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

#include "include/deploy/engine/tensorrt_engine.h"

namespace Deploy {


void TensorRTInferenceEngine::Init(std::string model_dir, uint32_t max_workspace_size, uint32_t max_batch_size, std::string trt_cache_filebool="", TensorRTInferenceConfigs configs}) {
  auto builder = InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(configs.logger_));
  
  const auto explicitBatch = max_bath_size << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = InferUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
  auto config = InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  auto parser = InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, configs.logger_));
  auto parsed = parser->parseFromFile(model_dir.c_str(), static_cast<int>(logger.mReportableSeverity)); 
  config->setMaxWorkspaceSize(max_workspace_size)
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), InferDeleter()); 
}

  void TensorRTInferenceEngine::Infer(const std::vector<DataBlob> &input_blobs,
	     std::vector<DataBlob> *output_blobs){
  
  
  }
}  //  namespace Deploy
