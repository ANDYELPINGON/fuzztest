// Copyright 2022 The Centipede Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "./centipede/corpus_io.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "./centipede/blob_file.h"
#include "./centipede/defs.h"
#include "./centipede/feature.h"
#include "./centipede/logging.h"
#include "./centipede/remote_file.h"
#include "./centipede/rusage_profiler.h"
#include "./centipede/util.h"

namespace centipede {

void ReadShard(std::string_view corpus_path, std::string_view features_path,
               const std::function<void(ByteArray, FeatureVec)> &callback) {
  const bool good_corpus_path =
      !corpus_path.empty() && RemotePathExists(corpus_path);
  const bool good_features_path =
      !features_path.empty() && RemotePathExists(features_path);

  if (!good_corpus_path) {
    LOG(WARNING) << "Corpus file path empty or not found - returning: "
                 << corpus_path;
    return;
  }

  RPROF_THIS_FUNCTION_WITH_TIMELAPSE(            //
      /*enable=*/VLOG_IS_ON(10),                 //
      /*timelapse_interval=*/absl::Seconds(30),  //
      /*also_log_timelapses=*/false);

  // Maps input hashes to inputs.
  absl::flat_hash_map<std::string /*hash*/, ByteArray /*input*/> hash_to_input;

  // Read inputs from the corpus file into `hash_to_input`.
  auto corpus_reader = DefaultBlobFileReaderFactory();
  CHECK_OK(corpus_reader->Open(corpus_path)) << VV(corpus_path);
  ByteSpan blob;
  while (corpus_reader->Read(blob).ok()) {
    hash_to_input.emplace(Hash(blob), ByteArray{blob.begin(), blob.end()});
  }

  const size_t num_all_inputs = hash_to_input.size();
  size_t num_inputs_with_good_features = 0;
  size_t num_inputs_with_empty_features = 0;
  size_t num_inputs_with_no_features = num_all_inputs;

  RPROF_SNAPSHOT("Read inputs");

  // If the features file is not passed or doesn't exist, simply ignore it.
  if (!good_features_path) {
    LOG(WARNING) << "Features file path empty or not found - ignoring: "
                 << features_path;
  } else {
    // Read features from the features file. For each feature, find a matching
    // input in `hash_to_input`, call `callback` for the pair, and remove the
    // entry from `hash_to_input`. In the end, `hash_to_input` will contain
    // only inputs without matching features.
    auto features_reader = DefaultBlobFileReaderFactory();
    CHECK_OK(features_reader->Open(features_path)) << VV(features_path);
    ByteSpan hash_and_features;
    while (features_reader->Read(hash_and_features).ok()) {
      // Every valid feature record must contain the hash at the end.
      // Ignore this record if it is too short.
      if (hash_and_features.size() < kHashLen) continue;
      FeatureVec features;
      std::string hash = UnpackFeaturesAndHash(hash_and_features, &features);
      auto input_it = hash_to_input.find(hash);
      if (input_it != hash_to_input.end()) {
        --num_inputs_with_no_features;
        if (features.empty()) {
          // When the features file got created, Centipede did compute features
          // for the input, but they came up empty. Indicate to the client that
          // there is no need to recompute by passing this special value.
          features = {feature_domains::kNoFeature};
          ++num_inputs_with_empty_features;
        } else {
          ++num_inputs_with_good_features;
        }
        callback(std::move(input_it->second), std::move(features));
        hash_to_input.erase(input_it);
      }
    }

    RPROF_SNAPSHOT("Read features & reported input/features pairs");
  }

  // Finally, call `callback` on the remaining inputs without matching features.
  // This also automatically covers the features file not passed or missing.
  for (auto &&[hash, input] : hash_to_input) {
    // Indicate to the client that it needs to recompute features for this input
    // by passing an empty value.
    callback(std::move(input), {});
  }

  RPROF_SNAPSHOT("Reported inputs with no matching features");

  VLOG(1)  //
      << "Finished shard reading:\n"
      << "Corpus path             : " << corpus_path << "\n"
      << "Features path           : " << features_path << "\n"
      << "Inputs                  : " << num_inputs_with_good_features << "\n"
      << "Inputs w/ good features : " << num_inputs_with_good_features << "\n"
      << "Inputs w/ empty features: " << num_inputs_with_empty_features << "\n"
      << "Inputs w/ no features   : " << num_inputs_with_no_features;
}

}  // namespace centipede
