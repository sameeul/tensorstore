
#include "tensorstore/driver/ometiff/metadata.h"

// ToDo - Clean up headers
#include <optional>

#include "absl/algorithm/container.h"
#include "absl/base/internal/endian.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "tensorstore/codec_spec_registry.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/data_type_endian_conversion.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/internal/json_binding/data_type.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/json_metadata_matching.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/json_bindable.h"

namespace tensorstore {
namespace internal_ometiff {
using internal::MetadataMismatchError;
namespace jb = tensorstore::internal_json_binding;
namespace {

absl::Status ValidateMetadata(OmeTiffMetadata& metadata) {
  return absl::OkStatus();
}

constexpr auto MetadataJsonBinder = [](auto maybe_optional) {
  return [=](auto is_loading, const auto& options, auto* obj, auto* j) {
    using T = internal::remove_cvref_t<decltype(*obj)>;
    DimensionIndex* rank = nullptr;
    if constexpr (is_loading) {
      rank = &obj->rank;
    }
    return jb::Object(
        jb::Member(
            "dimensions",
            jb::Projection(&T::shape, maybe_optional(jb::ShapeVector(rank)))),
        jb::Member("blockSize",
                   jb::Projection(&T::chunk_shape,
                                  maybe_optional(jb::ChunkShapeVector(rank)))),
        jb::Member(
            "dataType",
            jb::Projection(&T::dtype, maybe_optional(jb::Validate(
                                          [](const auto& options, auto* obj) {
                                            return absl::OkStatus();
                                          },
                                          jb::DataTypeJsonBinder))))

                    )(is_loading, options, obj, j);
  };
};

} //namespace

std::string OmeTiffMetadata::GetCompatibilityKey() const {
    // need to figure out what goes here
    ::nlohmann::json::object_t obj;
    span<const Index> chunk_shape = chunk_layout.shape();
    obj.emplace("blockSize", ::nlohmann::json::array_t(chunk_shape.begin(),
                                                        chunk_shape.end()));
    obj.emplace("dataType", dtype.name());
    return ::nlohmann::json(obj).dump();
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(OmeTiffMetadata, 
                              jb::Validate([](const auto& options, auto* obj) 
                              { return ValidateMetadata(*obj); },
                             MetadataJsonBinder(internal::identity{})))

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(OmeTiffMetadataConstraints,
                                       MetadataJsonBinder([](auto binder) {
                                         return jb::Optional(binder);
                                       }))



Result<std::shared_ptr<const OmeTiffMetadata>> GetNewMetadata(
    const OmeTiffMetadataConstraints& metadata_constraints, const Schema& schema) {
  auto metadata = std::make_shared<OmeTiffMetadata>();

  return metadata;
}

absl::Status SetChunkLayoutFromMetadata(
    DimensionIndex rank, std::optional<span<const Index>> chunk_shape,
    ChunkLayout& chunk_layout) {

// ToDo - Need to understand and reimplement 
  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(RankConstraint{rank}));
  rank = chunk_layout.rank();
  if (rank == dynamic_rank) return absl::OkStatus();

  {
    DimensionIndex inner_order[kMaxRank];
    for (DimensionIndex i = 0; i < rank; ++i) {
      inner_order[i] = rank - i - 1;
    }
    TENSORSTORE_RETURN_IF_ERROR(
        chunk_layout.Set(ChunkLayout::InnerOrder(span(inner_order, rank))));
  }
  if (chunk_shape) {
    assert(chunk_shape->size() == rank);
    TENSORSTORE_RETURN_IF_ERROR(
        chunk_layout.Set(ChunkLayout::ChunkShape(*chunk_shape)));
  }
  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(
      ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(rank))));
  return absl::OkStatus();
}

Result<SharedArrayView<const void>> DecodeChunk(const OmeTiffMetadata& metadata,
                                                absl::Cord buffer) {

    SharedArrayView<void> full_decoded_array(
      internal::AllocateAndConstructSharedElements(
          metadata.chunk_layout.num_elements(), value_init, metadata.dtype),
      metadata.chunk_layout);
  return full_decoded_array;
}

Result<absl::Cord> EncodeChunk(span<const Index> chunk_indices,
                               const OmeTiffMetadata& metadata,
                               ArrayView<const void> array) {

                                return absl::Cord();
}

Result<IndexDomain<>> GetEffectiveDomain(
    DimensionIndex rank, std::optional<span<const Index>> shape,
    const Schema& schema) {
  auto domain = schema.domain();
  if (!shape && !domain.valid()) {
    if (schema.rank() == 0) return {std::in_place, 0};
    // No information about the domain available.
    return {std::in_place};
  }

  // Rank is already validated by caller.
  assert(RankConstraint::EqualOrUnspecified(schema.rank(), rank));
  IndexDomainBuilder builder(std::max(schema.rank().rank, rank));
  if (shape) {
    builder.shape(*shape);
    builder.implicit_upper_bounds(true);
  } else {
    builder.origin(GetConstantVector<Index, 0>(builder.rank()));
  }

  TENSORSTORE_ASSIGN_OR_RETURN(auto domain_from_metadata, builder.Finalize());
  TENSORSTORE_ASSIGN_OR_RETURN(domain,
                               MergeIndexDomains(domain, domain_from_metadata),
                               tensorstore::MaybeAnnotateStatus(
                                   _, "Mismatch between metadata and schema"));
  return WithImplicitDimensions(domain, false, true);
  return domain;
}

Result<IndexDomain<>> GetEffectiveDomain(
    const OmeTiffMetadataConstraints& metadata_constraints, const Schema& schema) {
  return GetEffectiveDomain(metadata_constraints.rank,
                            metadata_constraints.shape, schema);
}
Result<ChunkLayout> GetEffectiveChunkLayout(
    DimensionIndex rank, std::optional<span<const Index>> chunk_shape,
    const Schema& schema) {
  auto chunk_layout = schema.chunk_layout();
  TENSORSTORE_RETURN_IF_ERROR(
      SetChunkLayoutFromMetadata(rank, chunk_shape, chunk_layout));
  return chunk_layout;
}

Result<ChunkLayout> GetEffectiveChunkLayout(
    const OmeTiffMetadataConstraints& metadata_constraints, const Schema& schema) {
  assert(RankConstraint::EqualOrUnspecified(metadata_constraints.rank,
                                            schema.rank()));
  return GetEffectiveChunkLayout(
      std::max(metadata_constraints.rank, schema.rank().rank),
      metadata_constraints.chunk_shape, schema);
}
}  // namespace internal_ometiff
}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_ometiff::OmeTiffMetadataConstraints,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::internal_ometiff::OmeTiffMetadataConstraints>())