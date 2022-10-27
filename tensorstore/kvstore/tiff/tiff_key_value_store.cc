#include <tiffio.h>
#include <regex>

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/context_binding.h"
#include "tensorstore/internal/file_io_concurrency_resource.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/os_error_code.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/file/unique_handle.h"
#include "tensorstore/kvstore/file/util.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/url_registry.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

// Include these last to reduce impact of macros.
#include "tensorstore/kvstore/file/posix_file_util.h"
#include "tensorstore/kvstore/file/windows_file_util.h"

namespace tensorstore {
namespace {

namespace jb = tensorstore::internal_json_binding;

using ::tensorstore::internal::GetLastErrorCode;
using ::tensorstore::internal::GetOsErrorStatusCode;
using ::tensorstore::internal::OsErrorCode;
using ::tensorstore::internal::StatusFromOsError;
using ::tensorstore::internal_file_util::FileDescriptor;
using ::tensorstore::internal_file_util::FileInfo;
using ::tensorstore::internal_file_util::GetFileInfo;
using ::tensorstore::internal_file_util::IsKeyValid;
using ::tensorstore::internal_file_util::kLockSuffix;
using ::tensorstore::internal_file_util::LongestDirectoryPrefix;
using ::tensorstore::internal_file_util::UniqueFileDescriptor;
using ::tensorstore::kvstore::ReadResult;

auto& tiff_bytes_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/tiff/bytes_read",
    "Bytes read by the tiff kvstore driver");

auto& tiff_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/tiff/read", "tiff driver kvstore::Read calls");

absl::Status ValidateKey(std::string_view key) {
  if (!IsKeyValid(key, kLockSuffix)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid key: ", tensorstore::QuoteString(key)));
  }
  return absl::OkStatus();
}

// Encode in the generation fields that uniquely identify the file.
StorageGeneration GetFileGeneration(const FileInfo& info) {
  return StorageGeneration::FromValues(internal_file_util::GetDeviceId(info),
                                       internal_file_util::GetFileId(info),
                                       internal_file_util::GetMTime(info));
}

/// Returns a absl::Status for the current errno value. The message is composed
/// by catenation of the provided string parts.
absl::Status StatusFromErrno(std::string_view a = {}, std::string_view b = {},
                             std::string_view c = {}, std::string_view d = {}) {
  return StatusFromOsError(GetLastErrorCode(), a, b, c, d);
}

absl::Status VerifyRegularFile(FileDescriptor fd, FileInfo* info,
                               const char* path) {
  if (!internal_file_util::GetFileInfo(fd, info)) {
    return StatusFromErrno("Error getting file information: ", path);
  }
  if (!internal_file_util::IsRegularFile(*info)) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("Not a regular file: ", path));
  }
  return absl::OkStatus();
}

Result<UniqueFileDescriptor> OpenValueFile(const char* path,
                                           StorageGeneration* generation,
                                           std::int64_t* size = nullptr) {
  UniqueFileDescriptor fd =
      internal_file_util::OpenExistingFileForReading(path);
  if (!fd.valid()) {
    auto error = GetLastErrorCode();
    if (GetOsErrorStatusCode(error) == absl::StatusCode::kNotFound) {
      *generation = StorageGeneration::NoValue();
      return fd;
    }
    return StatusFromOsError(error, "Error opening file: ", path);
  }
  FileInfo info;
  TENSORSTORE_RETURN_IF_ERROR(VerifyRegularFile(fd.get(), &info, path));
  if (size) *size = internal_file_util::GetSize(info);
  *generation = GetFileGeneration(info);
  return fd;
}

std::string GetDataType(short sample_format, short bits_per_sample){
  switch (sample_format) {
    case 1 :
      switch (bits_per_sample) {
        case 8:return "uint8";
          break;
        case 16:return "uint16";
          break;
        case 32:return "uint32";
          break;
        case 64:return "uint64";
          break;
        default: return "uint16";
      }
      break;
    case 2:
      switch (bits_per_sample) {
        case 8:return "int8";
          break;
        case 16:return "int16";
          break;
        case 32:return "int32";
          break;
        case 64:return "int64";
          break;
        default: return "uint16";
      }
      break;
    case 3:
      switch (bits_per_sample) {
        case 8:
        case 16:
        case 32:
          return "float";
          break;
        case 64:
          return "double";
          break;
        default: return "uint16";
      }
      break;
    default: return "uint16";
  }
}

/// Implements `TiffKeyValueStore::Read`.

// if we can override this in each cache class, that may work
struct ReadTask {
  std::string full_path;
  kvstore::ReadOptions options;

  Result<ReadResult> operator()() const {
    ReadResult read_result;
    std::string image_metadata;
    std::string tag = "/__TAG__/";
    std::string img_tag = "IMAGE_DESCRIPTION"; 
    std::size_t pos = full_path.rfind(tag);
    std::string actual_full_path = full_path.substr(0, pos);
    std::string tag_value = full_path.substr(pos+tag.length());

// need to make sure fd has the correct timestamp for stale check
    read_result.stamp.time = absl::Now();
    std::int64_t size;
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto fd,
        OpenValueFile(actual_full_path.c_str(), &read_result.stamp.generation, &size));
    if (!fd.valid()) {
      read_result.state = ReadResult::kMissing;
      return read_result;
    }
    if (read_result.stamp.generation == options.if_not_equal ||
        (!StorageGeneration::IsUnknown(options.if_equal) &&
         read_result.stamp.generation != options.if_equal)) {
      return read_result;
    }

    if (pos != std::string::npos){
      if (tag_value == img_tag){
        std::ostringstream oss;
        TIFF *tiff_ = TIFFOpen(actual_full_path.c_str(), "r");
        if (tiff_ != nullptr) 
        {
          read_result.state = ReadResult::kValue;
          char* infobuf;
          uint32_t 
            image_width = 0, 
            image_height = 0, 
            tile_width = 0,
            tile_height = 0;
          short
            sample_format = 0,          
            bits_per_sample = 0;

          TIFFGetField(tiff_, TIFFTAG_IMAGEWIDTH, &image_width);
          TIFFGetField(tiff_, TIFFTAG_IMAGELENGTH, &image_height);
          TIFFGetField(tiff_, TIFFTAG_TILEWIDTH, &tile_width);
          TIFFGetField(tiff_, TIFFTAG_TILELENGTH, &tile_height);
          TIFFGetField(tiff_, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
          TIFFGetField(tiff_, TIFFTAG_SAMPLEFORMAT, &sample_format);
          
          std::string dtype = GetDataType(sample_format, bits_per_sample);

          oss << "{"
              << "\"dimensions\": [" << image_height << "," << image_width << "],"
              << "\"blockSize\": [" << tile_height << "," << tile_width << "],"
              << "\"dataType\": \"" << dtype << "\""
              << "}";

        }
        TIFFClose(tiff_);      
        absl::Cord tmp =  absl::Cord(oss.str());
        read_result.value = std::move(tmp);
      }

      else // parse tile indices
      { 
        std::smatch match_result;
        std::regex tile_indices_regex("_(\\d+)_(\\d+)");
        if (regex_match(tag_value, match_result, tile_indices_regex)){
          uint32_t x_pos = std::stoi(match_result[2].str());
          uint32_t y_pos = std::stoi(match_result[1].str());
          TIFF *tiff_ = TIFFOpen(actual_full_path.c_str(), "r");
          if (tiff_ != nullptr) 
          {
            read_result.state = ReadResult::kValue;
            auto t_szb = TIFFTileSize(tiff_);
            internal::FlatCordBuilder buffer(t_szb);
            auto errcode = TIFFReadTile(tiff_, buffer.data(), x_pos, y_pos, 0, 0);
            TIFFClose(tiff_);      
            if (errcode != -1){
              tiff_bytes_read.IncrementBy(errcode);
              read_result.value = std::move(buffer).Build();
            } 
            else {
              return StatusFromErrno("Error reading file: ", actual_full_path);
            }
          }
        }
      }
    }
  
    return read_result;
  }
};

struct TiffKeyValueStoreSpecData {
  Context::Resource<internal::FileIoConcurrencyResource> file_io_concurrency;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.file_io_concurrency);
  };

   constexpr static auto default_json_binder = jb::Object(jb::Member(
      internal::FileIoConcurrencyResource::id,
      jb::Projection<&TiffKeyValueStoreSpecData::file_io_concurrency>()));
};

class TiffKeyValueStoreSpec
    : public internal_kvstore::RegisteredDriverSpec<TiffKeyValueStoreSpec,
                                                    TiffKeyValueStoreSpecData> {
 public:
  static constexpr char id[] = "tiff";

  Future<kvstore::DriverPtr> DoOpen() const override;

  Result<std::string> ToUrl(std::string_view path) const override {
    return tensorstore::StrCat(id, "://", internal::PercentEncodeUriPath(path));
  }
};

class TiffKeyValueStore
    : public internal_kvstore::RegisteredDriver<TiffKeyValueStore,
                                                TiffKeyValueStoreSpec> {
 public:
  Future<ReadResult> Read(Key key, ReadOptions options) override {
    tiff_read.Increment();
    TENSORSTORE_RETURN_IF_ERROR(ValidateKey(key));
    return MapFuture(executor(), ReadTask{std::move(key), std::move(options)});
  }

  const Executor& executor() { return spec_.file_io_concurrency->executor; }

  std::string DescribeKey(std::string_view key) override {
    return tensorstore::StrCat("local file ", tensorstore::QuoteString(key));
  }

  absl::Status GetBoundSpecData(TiffKeyValueStoreSpecData& spec) const {
    spec = spec_;
    return absl::OkStatus();
  }

  SpecData spec_;
};

Future<kvstore::DriverPtr> TiffKeyValueStoreSpec::DoOpen() const {
  auto driver_ptr = internal::MakeIntrusivePtr<TiffKeyValueStore>();
  driver_ptr->spec_ = data_;
  return driver_ptr;
}

Result<kvstore::Spec> ParseFileUrl(std::string_view url) {
  auto driver_spec = internal::MakeIntrusivePtr<TiffKeyValueStoreSpec>();
  driver_spec->data_.file_io_concurrency =
      Context::Resource<internal::FileIoConcurrencyResource>::DefaultSpec();
  auto parsed = internal::ParseGenericUri(url);
  assert(parsed.scheme == tensorstore::TiffKeyValueStoreSpec::id);
  if (!parsed.query.empty()) {
    return absl::InvalidArgumentError("Query string not supported");
  }
  if (!parsed.fragment.empty()) {
    return absl::InvalidArgumentError("Fragment identifier not supported");
  }
  return {std::in_place, std::move(driver_spec),
          internal::PercentDecode(parsed.authority_and_path)};
}

}  // namespace
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::TiffKeyValueStore)

namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::TiffKeyValueStoreSpec>
    registration;

const tensorstore::internal_kvstore::UrlSchemeRegistration
    url_scheme_registration{tensorstore::TiffKeyValueStoreSpec::id,
                            tensorstore::ParseFileUrl};
}  // namespace