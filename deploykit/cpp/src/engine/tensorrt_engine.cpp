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

void TensorRTInferenceEngine::Init(std::string model_dir,
        int max_workspace_size,
        int max_batch_size,
        std::string trt_cache_file,
        TensorRTInferenceConfigs configs) {
  auto builder = InferUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(configs.logger_));
  auto config =
      InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

  auto profile = builder->createOptimizationProfile();
  profile->setDimensions("x", nvinfer1::OptProfileSelector::kMIN,
                         nvinfer1::Dims4{1, 3, 640, 640});
  profile->setDimensions("x",  nvinfer1::OptProfileSelector::kOPT,
                          nvinfer1::Dims4{1, 3, 640, 640});
  profile->setDimensions("x",  nvinfer1::OptProfileSelector::kMAX,
                          nvinfer1::Dims4{1, 3, 640, 640});
  config->addOptimizationProfile(profile);

  config->setMaxWorkspaceSize(max_workspace_size);

  const auto explicitBatch =
      max_batch_size << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  auto network = InferUniquePtr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(explicitBatch));

  auto parser = InferUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, configs.logger_));
  auto parsed = parser->parseFromFile(
      model_dir.c_str(), static_cast<int>(configs.logger_.mReportableSeverity));
  
  engine_ =
      std::shared_ptr<nvinfer1::ICudaEngine>(
          builder->buildEngineWithConfig(*network, *config), InferDeleter());
}


 void TensorRTInferenceEngine::FeedInput(const std::vector<DataBlob>& input_blobs, const TensorRT::BufferManager& buffers){
  for(auto input_blob : input_blobs) {
    if (input_blob.dtype == 0) {
      float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(input_blob.name)); 
      hostDataBuffer = (float*)(input_blob.data.data());
    } else if (input_blob.dtype == 1) {
      int64_t* hostDataBuffer = static_cast<int64_t*>(buffers.getHostBuffer(input_blob.name)); 
      hostDataBuffer = (int64_t*)(input_blob.data.data());
    } else if (input_blob.dtype == 2) {
      int* hostDataBuffer = static_cast<int*>(buffers.getHostBuffer(input_blob.name)); 
      hostDataBuffer = (int*)(input_blob.data.data());
    } else if (input_blob.dtype == 3) {
      uint8_t* hostDataBuffer = static_cast<uint8_t*>(buffers.getHostBuffer(input_blob.name)); 
      hostDataBuffer = (uint8_t*)(input_blob.data.data());
    }
  } 
 }

void TensorRTInferenceEngine::Infer(const std::vector<DataBlob> &input_blobs,
                                    const int batch_size,
                                    std::vector<DataBlob> *output_blobs) {

  auto context = InferUniquePtr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, 640, 640}});

  TensorRT::BufferManager buffers(engine_, batch_size, context.get());
  FeedInput(input_blobs, buffers);
  buffers.copyInputToDevice();
  bool status = context->executeV2(buffers.getDeviceBindings().data());
  buffers.copyOutputToHost();

}
}  //  namespace Deploy
