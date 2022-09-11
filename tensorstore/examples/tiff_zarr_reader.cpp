// Copyright 2020 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Extracts a slice of a volumetric dataset, outputtting it as a 2d jpeg image.
//
// extract_slice --output_file=/tmp/foo.jpg --input_spec=...

#include <stdint.h>

#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "tensorstore/tensorstore.h"
#include "tensorstore/context.h"
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/compression/blosc.h"
//#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/open.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

using ::tensorstore::Context;
using ::tensorstore::StrCat;
using tensorstore::Index;


void read_ometiff_data()
{
  tensorstore::Context context = Context::Default();
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, tensorstore::Open({{"driver", "ometiff"},
                            {"kvstore", {{"driver", "file"},
                                         {"path", "/mnt/hdd8/axle/data/nyxus_zarr_test/eastman-plate01-intensity/eastman-plate01-intensity/p01_x01_y01_wx2_wy2_c1.ome.tif"}}
                            }},
                            context,
                            tensorstore::OpenMode::open,
                            tensorstore::RecheckCached{false},
                            tensorstore::ReadWriteMode::read).result());



 
}

int main(int argc, char** argv) {
  read_ometiff_data();
 return 0;
}