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

#include "include/deploy/engine/triton_engine.h"
#include "rapidjson/error/en.h"

namespace nic = nvidia::inferenceserver::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    nic::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }


namespace Deploy {

std::string  DtypeToString(long int dtype) {
    if (dtype == 0) {
      return "FP32";
    } else if (dtype == 1) {
      return "INT64";
    } else if (dtype == 2) {
      return "INT32";
    } else if (dtype == 3) {
      return "UINT8";
    }
}

int  DtypeToInt(std::string dtype) {
    if (dtype == "FP32") {
      return 0;
    } else if (dtype == "INT64") {
      return 1;
    } else if (dtype == "INT32") {
      return 2;
    } else if (dtype == "UINT8") {
      return 3;
    }
}

void TritonInferenceEngine::Init(const std::string& url, bool verbose) {
  FAIL_IF_ERR(nic::InferenceServerHttpClient::Create(
        &client_, url, verbose),
        "error: unable to create client for inference.")
}

nic::Error TritonInferenceEngine::GetModelMetaData(
        const std::string& model_name,
        const std::string& model_version,
        const nic::Headers& headers,
        rapidjson::Document* model_metadata) {
  std::string model_metadata_str;
  FAIL_IF_ERR(client_->ModelMetadata(
      &model_metadata_str, model_name, model_version, headers),
          "error: failed to get model metadata.");
  model_metadata->Parse(model_metadata_str.c_str(), model_metadata_str.size());
  if (model_metadata->HasParseError()) {
    return  nic::Error(
        "failed to parse JSON at" +
        std::to_string(model_metadata->GetErrorOffset()) +
        ": " + std::string(GetParseError_En(model_metadata->GetParseError())));
  }
  return nic::Error::Success;
}

void  TritonInferenceEngine::CreateInput(const std::vector<DataBlob> &input_blobs, std::vector<nic::InferInput*>* inputs){
  for (int i = 0; i < input_blobs.size(); i++) {
      nic::InferInput* input;
      std::vector<int64_t> input_shape = input_blobs[i].GetShape<int64_t>();
    for (int j = 0; j < input_shape.size(); j++){
        std::cout << input_shape[j] << std::endl;
    }
      nic::InferInput::Create(
              &input,
              input_blobs[i].name,
              input_blobs[i].GetShape<int64_t>(),
              DtypeToString(input_blobs[i].dtype));

      FAIL_IF_ERR(
              input->AppendRaw(reinterpret_cast<const uint8_t*>(&input_blobs[i].data[0]), input_blobs[i].data.size()),
              "error: unable to set data for INPUT.");
      inputs->push_back(input);
  }
}

void TritonInferenceEngine::CreateOutput(const rapidjson::Document& model_metadata, std::vector<const nic::InferRequestedOutput*>* outputs){
    const auto& output_itr = model_metadata.FindMember("outputs");
    size_t output_count = 0;
    for (rapidjson::Value::ConstValueIterator itr = output_itr->value.Begin();
            itr != output_itr->value.End();
            ++itr) {
      auto output_name = (*itr)["name"].GetString();
      nic::InferRequestedOutput* output;
      nic::InferRequestedOutput::Create(&output, output_name);
      outputs->push_back(std::move(output));
    }
}

void TritonInferenceEngine::ParseConfigs(const TritonInferenceConfigs &configs, nic::InferOptions* options){
    options->model_version_ = configs.model_version_;
    options->request_id_ = configs.request_id_;
    options->sequence_id_ = configs.sequence_id_;
    options->sequence_start_ = configs.sequence_start_;
    options->sequence_end_ = configs.sequence_end_;
    options->priority_ = configs.priority_;
    options->server_timeout_ = configs.server_timeout_;
    options->client_timeout_ = configs.client_timeout_;

}

void TritonInferenceEngine::Infer(const TritonInferenceConfigs& configs, const std::vector<DataBlob>& input_blobs, std::vector<DataBlob> *output_blobs, const nic::Headers& headers, const nic::Parameters& query_params){
  nic::InferOptions options(configs.model_name_);
  ParseConfigs(configs, &options);

  rapidjson::Document model_metadata;
  GetModelMetaData(
          options.model_name_,
          options.model_version_,
          headers,
          &model_metadata);

  std::vector<nic::InferInput*> inputs;
  CreateInput(input_blobs, &inputs);

  std::vector<const nic::InferRequestedOutput*> outputs;
  CreateOutput(model_metadata, &outputs);

  nic::InferResult* results;
  client_->Infer(&results, options, inputs, outputs, headers, query_params);

 for (const auto output: outputs) {
     std::string output_name = output->Name();

     DataBlob output_blob;
     output_blob.name = output->Name();

     std::vector<int64_t> output_shape;
     results->Shape(output->Name(), &output_shape);
     output_blob.SetShape(output_shape);

     std::string output_dtype;
     results->Datatype(output->Name(), &output_dtype);
     output_blob.dtype = DtypeToInt(output_dtype);

     // TODO: set output.lod when batch_size >1; 
     
     int size = 1;
     for (const auto& i : output_blob.shape) {
         size *= i;
     }
     size_t output_byte_size;
     uint8_t* output_data;
     results->RawData(output_blob.name, (const uint8_t**) &output_data, &output_byte_size);

     if (output_blob.dtype == 0) {
         output_blob.data.resize(size * sizeof(float));
     }
     else if (output_blob.dtype == 1) {
         output_blob.data.resize(size * sizeof(int64_t));
     }
     else if (output_blob.dtype == 2) {
         output_blob.data.resize(size * sizeof(int));
     }
     else if (output_blob.dtype == 3) {
         output_blob.data.resize(size * sizeof(uint8_t));
     }

     memcpy(output_blob.data.data(), output_data, size*sizeof(float));
     output_blobs->push_back(output_blob);
  }

}

}
