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

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <map>

#include "NvInfer.h"

#include "include/deploy/common/blob.h"
#include "include/deploy/engine/tensorrt/buffers.h"

namespace Deploy {

using InferUniquePtr = std::unique_ptr<T, InferDeleter>;
using Severity = nvinfer1::ILogger::Severity;

struct InferDeleter {
  template <typename T> void operator()(T *obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

struct TensorRTInferenceConfigs {
  void AnalysisConfig::SetTRTDynamicShapeInfo(
      std::map<std::string, std::vector<int>> min_input_shape,
      std::map<std::string, std::vector<int>> max_input_shape,
      std::map<std::string, std::vector<int>> optim_input_shape) {
    min_input_shape_ = min_input_shape;
    max_input_shape_ = max_input_shape;
    optim_input_shape_ = optim_input_shape;
  }

 protected:
  nvinfer1::ILogger &logger nvinfer1::ILogger &logger = NaiveLogger.Global();
  std::map<std::string, std::vector<int>> min_input_shape_{};
  std::map<std::string, std::vector<int>> max_input_shape_{};
  std::map<std::string, std::vector<int>> optim_input_shape_{};
};

// A logger for create TensorRT infer builder.
class NaiveLogger : public nvinfer1::ILogger {
 public:
  explicit NaiveLogger(Severity severity = Severity::kWARNING)
      : mReportableSeverity(severity) {}

  void log(nvinfer1::ILogger::Severity severity, const char *msg) override {
    switch (severity) {
    case Severity::kINFO:
      VLOG(3) << msg;
      break;
    case Severity::kWARNING:
      LOG(WARNING) << msg;
      break;
    case Severity::kINTERNAL_ERROR:
    case Severity::kERROR:
      LOG(ERROR) << msg;
      break;
    default:
      break;
    }
  }

  static nvinfer1::ILogger &Global() {
    static nvinfer1::ILogger *x = new NaiveLogger;
    return *x;
  }

  ~NaiveLogger() override {}

 private:
  Severity mReportableSeverity;
};

class TensorRTInferenceEngine {
 public:
  void Init(std::string model_dir, int max_workspace_size, int max_batch_size,
            std::string trt_cache_filebool = "",
            TensorRTInferenceConfigs configs = TensorRTInferenceConfigs());

  void Infer(const std::vector<DataBlob> &input_blobs,
             std::vector<DataBlob> *output_blobs);

  std::unique_ptr<nvinfer1::ICudaEngine> engine_;

 private:
  void ParseConfigs(const TritonInferenceConfigs &configs,
                    nic::InferOptions *options);

  void FeedInput(const std::vector<DataBlob> &input_blobs,
                 std::vector<nic::InferInput *> *inputs);

  void FetchOutput(c std::vector<const nic::InferRequestedOutput *> *outputs);

  void ParseONNXModel(const std::string &model_dir);
};

}  //  namespace Deploy
