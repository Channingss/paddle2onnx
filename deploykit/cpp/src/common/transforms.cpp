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


#include <math.h>

#include <iostream>
#include <string>
#include <vector>

#include "include/deploy/common/transforms.h"

namespace Deploy {

bool Normalize::Run(cv::Mat *im) {
  std::vector<float> range_val;
  for (int c = 0; c < im->channels(); c++) {
    range_val.push_back(max_val_[c] - min_val_[c]);
  }
  std::vector<cv::Mat> split_im;
  cv::split(*im, split_im);
  for (int c = 0; c < im->channels(); c++) {
    cv::subtract(split_im[c], cv::Scalar(min_val_[c]), split_im[c]);
    if (is_scale_) {
      float range_val = max_val_[c] - min_val_[c];
      cv::divide(split_im[c], cv::Scalar(range_val), split_im[c]);
    }
    cv::subtract(split_im[c], cv::Scalar(mean_[c]), split_im[c]);
    cv::divide(split_im[c], cv::Scalar(std_[c]), split_im[c]);
  }
  cv::merge(split_im, *im);
  return true;
}

bool Normalize::ShapeInfer(ShapeInfo* shape_info) {
  std::vector<int> before_shape = shape_info->shape.back();
  shape_info->transform_order.push_back("Normalize");
  shape_info->shape.push_back(before_shape);
  return true;
}


float ResizeByShort::GenerateScale(const int origin_w, const int origin_h) {
  int im_size_max = std::max(origin_w, origin_h);
  int im_size_min = std::min(origin_w, origin_h);
  float scale =
      static_cast<float>(target_size_) / static_cast<float>(im_size_min);
  if (max_size_ > 0) {
    if (round(scale * im_size_max) > max_size_) {
      scale = static_cast<float>(max_size_) / static_cast<float>(im_size_max);
    }
  }
  return scale;
}

bool ResizeByShort::Run(cv::Mat *im) {
  int origin_w = im->cols;
  int origin_h = im->rows;
  float scale = GenerateScale(origin_w, origin_h);
  int width = static_cast<int>(round(scale * im->cols));
  int height = static_cast<int>(round(scale * im->rows));
  cv::resize(*im, *im, cv::Size(width, height), 0, 0, interp_);
  return true;
}

bool ResizeByShort::ShapeInfer(ShapeInfo* shape_info) {
  std::vector<int> before_shape = shape_info->shape.back();
  shape_info->transform_order.push_back("ResizelsByShort");
  float scale = GenerateScale(before_shape[0], before_shape[1]);
  int width = static_cast<int>(round(scale * before_shape[0]));
  int height = static_cast<int>(round(scale * before_shape[1]));
  std::vector<int> after_shape = {width, height};
  shape_info->shape.push_back(after_shape);
  return true;
}

float ResizeByLong::GenerateScale(const int origin_w, const int origin_h) {
  int im_size_max = std::max(origin_w, origin_h);
  int im_size_min = std::min(origin_w, origin_h);
  float scale = 1.0f;
  if (target_size_ == -1) {
    if (im_size_max > max_size_) {
      scale = static_cast<float>(max_size_) / static_cast<float>(im_size_max);
    }
  } else {
    scale = static_cast<float>(target_size_) / static_cast<float>(im_size_max);
  }
  return scale;
}

bool ResizeByLong::Run(cv::Mat *im) {
  int origin_w = im->cols;
  int origin_h = im->rows;
  float scale = GenerateScale(origin_w, origin_h);
  int width = static_cast<int>(round(scale * im->cols));
  int height = static_cast<int>(round(scale * im->rows));
  if (stride_ != 0) {
    if (width / stride_ < 1 + 1e-5) {
      width = stride_;
    } else {
      width = (width / 32) * 32;
    }
    if (height / stride_ < 1 + 1e-5) {
      height = stride_;
    } else {
      height = (height / 32) * 32;
    }
  }
  cv::resize(*im, *im, cv::Size(width, height), 0, 0, interp_);
  return true;
}

bool ResizeByLong::ShapeInfer(ShapeInfo* shape_info) {
  std::vector<int> before_shape = shape_info->shape.back();
  shape_info->transform_order.push_back("ResizeByLong");
  float scale = GenerateScale(before_shape[0], before_shape[1]);
  int width = static_cast<int>(round(scale * before_shape[0]));
  int height = static_cast<int>(round(scale * before_shape[1]));
  if (stride_ != 0) {
    if (width / stride_ < 1 + 1e-5) {
      width = stride_;
    } else {
      width = (width / 32) * 32;
    }
    if (height / stride_ < 1 + 1e-5) {
      height = stride_;
    } else {
      height = (height / 32) * 32;
    }
  }
  std::vector<int> after_shape = {width, height};
  shape_info->shape.push_back(after_shape);
  return true;
}


bool Resize::Run(cv::Mat *im) {
  if (width_ <= 0 || height_ <= 0) {
    std::cerr << "[Resize] width and height should be greater than 0"
              << std::endl;
    return false;
  }
  cv::resize(*im, *im, cv::Size(width_, height_), 0, 0, interp_);
  return true;
}

bool Resize::ShapeInfer(ShapeInfo* shape_info) {
  std::vector<int> before_shape = shape_info->shape.back();
  shape_info->transform_order.push_back("Resize");
  std::vector<int> after_shape = {width_, height_};
  shape_info->shape.push_back(after_shape);
  return true;
}

bool CenterCrop::Run(cv::Mat *im) {
  int height = static_cast<int>(im->rows);
  int width = static_cast<int>(im->cols);
  if (height < height_ || width < width_) {
    std::cerr << "[CenterCrop] Image size less than crop size" << std::endl;
    return false;
  }
  int offset_x = static_cast<int>((width - width_) / 2);
  int offset_y = static_cast<int>((height - height_) / 2);
  cv::Rect crop_roi(offset_x, offset_y, width_, height_);
  *im = (*im)(crop_roi);
  return true;
}

bool CenterCrop::ShapeInfer(ShapeInfo* shape_info) {
  std::vector<int> before_shape = shape_info->shape.back();
  shape_info->transform_order.push_back("CenterCrop");
  std::vector<int> after_shape = {width_, height_};
  shape_info->shape.push_back(after_shape);
  return true;
}

void Padding::GeneralPadding(cv::Mat* im,
                             const std::vector<float> &padding_val,
                             int padding_w, int padding_h) {
  cv::Scalar value;
  if (im->channels() == 1) {
    value = cv::Scalar(padding_val[0]);
  } else if (im->channels() == 2) {
    value = cv::Scalar(padding_val[0], padding_val[1]);
  } else if (im->channels() == 3) {
    value = cv::Scalar(padding_val[0], padding_val[1], padding_val[2]);
  } else if (im->channels() == 4) {
    value = cv::Scalar(padding_val[0], padding_val[1], padding_val[2],
                                  padding_val[3]);
  }
  cv::copyMakeBorder(
  *im,
  *im,
  0,
  padding_h,
  0,
  padding_w,
  cv::BORDER_CONSTANT,
  value);
}

void Padding::MultichannelPadding(cv::Mat* im,
                                  const std::vector<float> &padding_val,
                                  int padding_w, int padding_h) {
  std::vector<cv::Mat> padded_im_per_channel(im->channels());
  for (size_t i = 0; i < im->channels(); i++) {
    const cv::Mat per_channel = cv::Mat(im->rows + padding_h,
                                        im->cols + padding_w,
                                        CV_32FC1,
                                        cv::Scalar(padding_val[i]));
    padded_im_per_channel[i] = per_channel;
  }
  cv::Mat padded_im;
  cv::merge(padded_im_per_channel, padded_im);
  cv::Rect im_roi = cv::Rect(0, 0, im->cols, im->rows);
  im->copyTo(padded_im(im_roi));
  *im = padded_im;
}

bool Padding::Run(cv::Mat *im) {
  int padding_w = 0;
  int padding_h = 0;
  if (width_ > 1 & height_ > 1) {
    padding_w = width_ - im->cols;
    padding_h = height_ - im->rows;
  } else if (stride_ >= 1) {
    int h = im->rows;
    int w = im->cols;
    padding_h =
        ceil(h * 1.0 / stride_) * stride_ - im->rows;
    padding_w =
        ceil(w * 1.0 / stride_) * stride_ - im->cols;
  }

  if (padding_h < 0 || padding_w < 0) {
    std::cerr << "[Padding] Computed padding_h=" << padding_h
              << ", padding_w=" << padding_w
              << ", but they should be greater than 0." << std::endl;
    return false;
  }
  if (im->channels() < 5) {
    Padding::GeneralPadding(&*im, im_value_, padding_w, padding_h);
  } else {
    Padding::MultichannelPadding(
      &*im,
      im_value_,
      padding_w,
      padding_h);
  }
  return true;
}

bool Padding::Run(cv::Mat *im, int max_w, int max_h) {
  int padding_w = 0;
  int padding_h = 0;
  if ((max_w - im->cols) > 0 || (max_h - im->rows) > 0) {
    padding_w = max_w - im->cols;
    padding_h = max_h - im->rows;
    cv::Scalar value = cv::Scalar(0, 0, 0);
    cv::copyMakeBorder(
      *im,
      *im,
      0,
      padding_h,
      0,
      padding_w,
      cv::BORDER_CONSTANT,
      value);
  }
  return true;
}


bool Padding::ShapeInfer(ShapeInfo* shape_info) {
  std::vector<int> before_shape = shape_info->shape.back();
  shape_info->transform_order.push_back("Padding");
  std::vector<int> after_shape = {width_, height_};
  shape_info->shape.push_back(after_shape);
  return true;
}

bool Clip::Run(cv::Mat *im) {
  std::vector<cv::Mat> split_im;
  cv::split(*im, split_im);
  for (int c = 0; c < im->channels(); c++) {
    cv::threshold(split_im[c], split_im[c], max_val_[c], max_val_[c],
                  cv::THRESH_TRUNC);
    cv::subtract(cv::Scalar(0), split_im[c], split_im[c]);
    cv::threshold(split_im[c], split_im[c], min_val_[c], min_val_[c],
                  cv::THRESH_TRUNC);
    cv::divide(split_im[c], cv::Scalar(-1), split_im[c]);
  }
  cv::merge(split_im, *im);
  return true;
}

bool Clip::ShapeInfer(ShapeInfo* shape_info) {
  std::vector<int> before_shape = shape_info->shape.back();
  shape_info->transform_order.push_back("Clip");
  shape_info->shape.push_back(before_shape);
  return true;
}

bool BGR2RGB::Run(cv::Mat *im) {
  cv::cvtColor(*im, *im, cv::COLOR_BGR2RGB);
  return true;
}

bool BGR2RGB::ShapeInfer(ShapeInfo* shape_info) {
  std::vector<int> before_shape = shape_info->shape.back();
  shape_info->transform_order.push_back("BGR2RGB");
  shape_info->shape.push_back(before_shape);
  return true;
}

bool RGB2BGR::Run(cv::Mat *im) {
  cv::cvtColor(*im, *im, cv::COLOR_RGB2BGR);
  return true;
}

bool RGB2BGR::ShapeInfer(ShapeInfo* shape_info) {
  std::vector<int> before_shape = shape_info->shape.back();
  shape_info->transform_order.push_back("RGB2BGR");
  shape_info->shape.push_back(before_shape);
  return true;
}

bool Permute::Run(cv::Mat *im) {
  cv::Mat im_clone = (*im).clone();
  int rh = im_clone.rows;
  int rw = im_clone.cols;
  int rc = im_clone.channels();
  float *data = reinterpret_cast<float*>(im->data);
  for (int i = 0; i < rc; ++i) {
    cv::extractChannel(im_clone,
                      cv::Mat(rh, rw, CV_32FC1, data + i * rh * rw), i);
  }
  return true;
}

bool Permute::ShapeInfer(ShapeInfo* shape_info) {
  std::vector<int> before_shape = shape_info->shape.back();
  shape_info->transform_order.push_back("Permute");
  shape_info->shape.push_back(before_shape);
  return true;
}


bool Convert::Run(cv::Mat *im) {
  if (dtype_ == "float") {
    im->convertTo(*im, CV_32FC(im->channels()));
  }
  return true;
}

bool Convert::ShapeInfer(ShapeInfo* shape_info) {
  std::vector<int> before_shape = shape_info->shape.back();
  shape_info->transform_order.push_back("Convert");
  shape_info->shape.push_back(before_shape);
  return true;
}

int OcrResize::GeneralWidth(int w, int h) {
  int resize_w;
  float ratio = static_cast<float>(w) / static_cast<float>(h);
  if (!fix_width_) {
    width_ = static_cast<int>(32 * ratio);
  }
  if (ceilf(height_ * ratio) > width_) {
    resize_w = width_;
  } else {
    resize_w = static_cast<int>(ceilf(height_ * ratio));
  }
  return resize_w;
}

bool OcrResize::Run(cv::Mat *im) {
  int resize_w = GeneralWidth(im->cols, im->rows);
  cv::resize(*im, *im, cv::Size(resize_w, height_), 0.f, 0.f, interp_);
  if (resize_w < width_ || is_pad_) {
    cv::copyMakeBorder(*im, *im, 0, 0, 0,
                      static_cast<int>(width_ - resize_w),
                      cv::BORDER_CONSTANT, value_);
  }
  return true;
}

bool OcrResize::ShapeInfer(ShapeInfo* shape_info) {
  std::vector<int> before_shape = shape_info->shape.back();
  shape_info->transform_order.push_back("OcrResize");
  int resize_w = GeneralWidth(before_shape[0], before_shape[1]);
  if (resize_w < width_ || is_pad_) {
    resize_w = width_;
  }
  std::vector<int> after_shape = {resize_w, height_};
  shape_info->shape.push_back(after_shape);
  return true;
}

bool OcrTrtResize::Run(cv::Mat *im) {
  int k = static_cast<int>(im->cols * 32 / im->rows);
  if (k >= width_) {
    cv::resize(*im, *im, cv::Size(width_, height_), 0.f, 0.f, interp_);
  } else {
    cv::resize(*im, *im, cv::Size(k, height_),
                0.f, 0.f, cv::INTER_LINEAR);
    cv::copyMakeBorder(*im, *im, 0, 0, 0,
          static_cast<int>(width_ - k), cv::BORDER_CONSTANT, {127, 127, 127});
  }
  return true;
}

bool OcrTrtResize::ShapeInfer(ShapeInfo* shape_info) {
  shape_info->transform_order.push_back("OcrTrtResize");
  std::vector<int> after_shape = {width_, height_};
  shape_info->shape.push_back(after_shape);
  return true;
}


}  // namespace Deploy
