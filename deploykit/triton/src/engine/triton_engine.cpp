// Copyright (c) 2021 TritonTriton Authors. All Rights Reserved.
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

namespace nic = nvidia::inferenceserver::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    nic::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

std::string  DtypeToString(int dtype){                               
  {                                                                
    if (dtype == 0) {                                              
      return "FP32";                                               
    }                                                              
    else if (dtype == 1) {                                         
      return "INT64";                                              
    }                                                              
    else if (dtype == 2) {                                         
      return "INT32";                                               
    }                                                              
    else if (dtype == 3) {                                         
      return "UINT8";                                              
    }
 };
};
int  DtypeToInt(std::string dtype){                               
  {                                                                
    if (dtype == "FP32") {                                              
      return 0;                                               
    }                                                              
    else if (dtype == "INT64") {                                         
      return 1;                                              
    }                                                              
    else if (dtype == "INT32") {                                         
      return 2;                                               
    }                                                              
    else if (dtype == "UINT8") {                                         
      return 3;                                              
    }
 };
};
namespace Deploy {

void TritonInferenceEngine::Init(std::string model_dir, TritonInferenceConfig &engine_config) {
  nic::Error err = nic::InferenceServerHttpClient::Create(
        &client_, engine_config.url, engine_config.verbose);	
  std::cout << engine_config.url << std::endl;
  if (!err.IsOk()) {
    std::cerr << "error: unable to create client for inference: " << err
              << std::endl;
    exit(1);
  };
};

void TritonInferenceEngine::Infer(std::vector<DataBlob> &input_blobs, std::vector<DataBlob> *output_blobs){
  nic::Headers http_headers;
  std::vector<nic::InferInput*> inputs;
  nic::Error err;
  for (int i = 0; i < input_blobs.size(); i++) {
      nic::InferInput* input;
      std::cout << DtypeToString(input_blobs[i].dtype) << std::endl;
      nic::InferInput::Create(&input, input_blobs[i].name, input_blobs[i].shape, DtypeToString(input_blobs[i].dtype));
      std::vector<uint8_t> data(input_blobs[i].data.begin(), input_blobs[i].data.end());
      std::cout << data.size() << std::endl;
      FAIL_IF_ERR(input->AppendRaw(data), "unable to set data for INPUT");
      inputs.push_back(input);
  };
 std::vector<const nic::InferRequestedOutput*> outputs = {};
 nic::InferRequestedOutput* output;
 nic::InferRequestedOutput::Create(&output, "save_infer_model/scale_0.tmp_0");
 std::shared_ptr<nic::InferRequestedOutput> output_ptr;
 output_ptr.reset(output);
 outputs.push_back(output_ptr.get());
 nic::InferOptions options("ppyolo");
 options.model_version_ = "";
 options.client_timeout_ = 0;
 nic::InferResult* results;
 err = client_->Infer(&results, options, inputs, outputs, http_headers);

 for (const auto output: outputs) {
     std::string output_name = output->Name(); 
     DataBlob output_blob;
     output_blob.name = output->Name();
     results->Shape(output->Name(), &output_blob.shape);
     std::string output_dtype;
     results->Datatype(output->Name(), &output_dtype);
     output_blob.dtype = DtypeToInt(output_dtype);
     //output.lod = output_tensor->lod();
     int size = 1;
     for (const auto& i : output_blob.shape) {
         size *= i;
     }
     size_t output_byte_size;
     char* output_data = output_blob.data.data();
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
     results->RawData(output_blob.name, (const uint8_t**)&output_data, &output_byte_size);
     output_blobs->push_back(std::move(output_blob));
     std::cout << "77" << std::endl;
  }
};

}
