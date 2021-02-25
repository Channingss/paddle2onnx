// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vector>
#include <string>
#include <iostream>
#include "http_client.h"
#include "common.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"

#include "include/deploy/common/blob.h"


#ifdef _WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif

namespace nic = nvidia::inferenceserver::client;

namespace Deploy {

class TritonInferenceEngine{
  public:
    void Init(const std::string& url, bool verbose=false);

    void Infer(const nic::InferOptions& options, const std::vector<DataBlob>& input_blobs, std::vector<DataBlob> *output_blobs, const nic::Headers& headers=nic::Headers(), const nic::Parameters& query_params=nic::Parameters());

    std::unique_ptr<nic::InferenceServerHttpClient> client_;

  private:

    void CreateInput(const std::vector<DataBlob> &input_blobs, std::vector<nic::InferInput*>* inputs);

    void CreateOutput(const rapidjson::Document& model_metadata, std::vector<const nic::InferRequestedOutput*>* outputs);

    nic::Error GetModelMetaData(const std::string& model_name, const std::string& model_version, const nic::Headers& http_headers, rapidjson::Document* model_metadata);
};

}
