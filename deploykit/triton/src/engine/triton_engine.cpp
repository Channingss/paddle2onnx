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

void TritonInferenceEngine::Init(std::string url, bool verbose=false) {
  FAIL_IF_ERR(nic::InferenceServerHttpClient::Create(
        &client_, url, verbose), unable to create client for inference)	
};

void TritonInferenceEngine::Infer(nic::InferOptions options, std::vector<DataBlob> &input_blobs, std::vector<DataBlob> *output_blobs){
  nic::Headers http_headers;
  std::vector<nic::InferInput*> inputs;
  nic::Error err;
  for (int i = 0; i < input_blobs.size(); i++) {
      nic::InferInput* input;
      std::cout << DtypeToString(input_blobs[i].dtype) << std::endl;
      nic::InferInput::Create(&input, input_blobs[i].name, input_blobs[i].shape, DtypeToString(input_blobs[i].dtype));

      std::vector<uint8_t> *data = new std::vector<uint8_t>;
      data->resize(input_blobs[i].data.size());
      memcpy(data->data(), input_blobs[i].data.data(), input_blobs[i].data.size());

      std::cout << data->size() << std::endl;
      FAIL_IF_ERR(input->AppendRaw(*data), "unable to set data for INPUT");
      inputs.push_back(input);
  };

    std::vector<const nic::InferRequestedOutput*> outputs = {};
    nic::InferRequestedOutput* output;
    nic::InferRequestedOutput::Create(&output, "save_infer_model/scale_0.tmp_0");
    std::shared_ptr<nic::InferRequestedOutput> output_ptr;
    output_ptr.reset(output);
    outputs.push_back(output_ptr.get());
    nic::InferResult* results;
    err = client_->Infer(&results, options, inputs, outputs, http_headers);

 for (const auto output: outputs) {
     std::string output_name = output->Name(); 
     DataBlob *output_blob = new DataBlob();
     output_blob->name = output->Name();
     results->Shape(output->Name(), &output_blob->shape);
     std::string output_dtype;
     results->Datatype(output->Name(), &output_dtype);
     output_blob->dtype = DtypeToInt(output_dtype);
     //output.lod = output_tensor->lod();
     int size = 1;
     for (const auto& i : output_blob->shape) {
         size *= i;
     }
     size_t output_byte_size;
     float* output_data; //= (float*)output_blob.data.data();

     results->RawData(output_blob->name, (const uint8_t**) &output_data, &output_byte_size);

     if (output_blob->dtype == 0) {
         output_blob->data.resize(size * sizeof(float));
     }
     else if (output_blob->dtype == 1) {
         output_blob->data.resize(size * sizeof(int64_t));
     }
     else if (output_blob->dtype == 2) {
         output_blob->data.resize(size * sizeof(int));
     }
     else if (output_blob->dtype == 3) {
         output_blob->data.resize(size * sizeof(uint8_t));
     }

     memcpy(output_blob->data.data(), output_data, size*sizeof(float));

     // output_blobs->push_back(std::move(*output_blob));
     float* tmp = (float*)output_blob->data.data();
     for (int i=0; i< 12; i++){
     if (i%6==0){
        std::cout << "box:"<< std::endl;
     }
     std::cout << tmp[i] << std::endl;
     }
  }
};

}
