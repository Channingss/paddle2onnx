# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.a

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import numpy as np

import cv2


class Transform(object):
    def __init__(self):
        pass

    def infer_shape(self):
        pass

    def __call_(self):
        pass


class Normalize(Transform):
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 min_val=[0, 0, 0],
                 max_val=[255., 255., 255.],
                 is_scale=True):
        self.op_name = self.__class__.__name__
        if not (isinstance(mean, Sequence) and isinstance(std, Sequence) \
            and isinstance(min_val, Sequence) and isinstance(max_val, Sequence)):
            raise TypeError(
                "[{}] Type of input mean/std/min_val/max_val should be list or tuple, but recieved is {}/{}/{}/{}.".
                format(self.op_name,
                       type(mean), type(std), type(min_val), type(max_val)))

        if not isinstance(is_scale, bool):
            raise TypeError(
                "[{}] Type of is_scale should be bool, but recieved is {}.".
                format(self.op_name, type(is_scale)))
        from functools import reduce
        if reduce(lambda x, y: x * y, std) == 0:
            raise ValueError(
                '[{}] There should not be 0 in std, but recieved is {}'.format(
                    self.op_name, std))
        if is_scale:
            if reduce(lambda x, y: x * y,
                      [a - b for a, b in zip(max_val, min_val)]) == 0:
                raise ValueError(
                    '[{}] There should not be 0 in (max_val - min_val), but recieved is {}'.
                    format(self.op_name, max_val - min_val))
        input_length_list = (len(mean), len(std), len(min_val), len(max_val))
        if len(set(input_length_list)) > 1:
            raise Exception(
                "[{}] Length of mean/std/min_val/max_val should be the same, but recieved is {}".
                format(self.op_name, input_length_list))

        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.is_scale = is_scale

    def __call__(self, im):
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :].astype(np.float32)
        std = np.array(self.std)[np.newaxis, np.newaxis, :].astype(np.float32)
        min_val = np.array(self.min_val)[np.newaxis, np.newaxis, :].astype(
            np.float32)
        max_val = np.array(self.max_val)[np.newaxis, np.newaxis, :].astype(
            np.float32)
        im -= min_val
        if self.is_scale:
            range_val = max_val - min_val
            im /= range_val
        im -= mean
        im /= std
        return im

    def infer_shape(self, shape_info):
        shape_info.shape[self.op_name] = shape_info.get_last_shape()
        return shape_info


class ResizeByShort(Transform):
    interp_list = [
        cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
        cv2.INTER_LANCZOS4
    ]

    def __init__(self, target_size, max_size=-1, interp=0):
        self.op_name = self.__class__.__name__
        if not isinstance(target_size, int):
            raise TypeError(
                "[{}] Type of target_size is invalid. Must be Integer, but recieved is {}"
                .format(self.op_name, type(target_size)))
        if not isinstance(max_size, int):
            raise TypeError(
                "[{}] Type of max_size is invalid. Must be Integer, but recieved is {}"
                .format(self.op_name, type(max_size)))
        if target_size <= 0:
            raise ValueError(
                "[{}] target_size should be greater than 0, but recieved target_size is {}".
                format(self.op_name, target_size))
        if max_size <= -2:
            raise ValueError(
                "[{}] max_size should not be less than -1, but recieved max_size is {}".
                format(self.op_name, max_size))

        if not isinstance(interp, int):
            raise TypeError(
                "[{}] Type of interp is invalid. Must be Integer, but recieved is {}"
                .format(self.op_name, type(interp)))
        if interp not in self.interp_list:
            raise ValueError(
                "[{}] Interp should be one of {}, but recieved is {}.".format(
                    self.op_name, self.interp_list, interp))

        self.target_size = target_size
        self.max_size = max_size
        self.interp = interp

    def _generate_scale(self, im_h, im_w):
        short_size = min(im_h, im_w)
        long_size = max(im_h, im_w)
        scale = float(self.target_size) / short_size
        if self.max_size > 0 and np.round(scale * long_size) > self.max_size:
            scale = float(self.max_size) / float(long_size)
        return scale

    def __call__(self, im):
        origin_height, origin_width, _ = im.shape
        scale = self._generate_scale(im_h, im_w)
        resized_height = int(round(origin_height * scale))
        resized_width = int(round(origin_width * scale))
        im = cv2.resize(
            im, (resized_width, resized_height), interpolation=self.interp)
        if im.ndim < 3:
            im = np.expand_dims(im, axis=-1)
        return im

    def infer_shape(self, shape_info):
        origin_height, origin_width = shape_info.get_last_shape()
        scale = self._generate_scale(origin_height, origin_width)
        resized_height = int(round(origin_height * scale))
        resized_width = int(round(origin_width * scale))
        shape_info.shape[self.op_name] = (resized_width, resized_height)
        return shape_info


class ResizeByLong(Transform):
    interp_list = [
        cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
        cv2.INTER_LANCZOS4
    ]

    def __init__(self, target_size, max_size=-1, interp=cv2.INTER_LINEAR):
        self.op_name = self.__class__.__name__
        if not isinstance(target_size, int):
            raise TypeError(
                "[{}] Type of target_size is invalid. Must be Integer, but recieved is {}"
                .format(self.op_name, type(target_size)))
        if not isinstance(max_size, int):
            raise TypeError(
                "[{}] Type of max_size is invalid. Must be Integer, but recieved is {}"
                .format(self.op_name, type(max_size)))
        if target_size <= 0:
            raise ValueError(
                "[{}] target_size should be greater than 0, but recieved target_size is {}".
                format(self.op_name, target_size))
        if max_size <= -2:
            raise ValueError(
                "[{}] max_size should not be less than -1, but recieved max_size is {}".
                format(self.op_name, max_size))

        if not isinstance(interp, int):
            raise TypeError(
                "[{}] Type of interp is invalid. Must be Integer, but recieved is {}"
                .format(self.op_name, type(interp)))
        if interp not in self.interp_list:
            raise ValueError(
                "[{}] Interp should be one of {}, but recieved is {}.".format(
                    self.op_name, self.interp_list, interp))

        self.target_size = target_size
        self.max_size = max_size
        self.interp = interp

    def _generate_scale(self, im_h, im_w):
        short_size = min(im_h, im_w)
        long_size = max(im_h, im_w)
        scale = float(self.target_size) / long_size
        if self.max_size > 0 and np.round(scale * short_size) > self.max_size:
            scale = float(self.max_size) / float(short_size)
        return scale

    def __call__(self, im):
        origin_height, origin_width, _ = im.shape
        scale = self._generate_scale(im_h, im_w)
        resized_height = int(round(origin_height * scale))
        resized_width = int(round(origin_width * scale))
        im = cv2.resize(
            im, (resized_width, resized_height), interpolation=self.interp)
        if im.ndim < 3:
            im = np.expand_dims(im, axis=-1)
        return im

    def infer_shape(self, shape_info):
        origin_height, origin_width = shape_info.get_last_shape()
        scale = self._generate_scale(origin_height, origin_width)
        resized_height = int(round(origin_height * scale))
        resized_width = int(round(origin_width * scale))
        shape_info.shape[self.op_name] = (resized_width, resized_height)
        return shape_info


class Resize(Transform):
    interp_list = [
        cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
        cv2.INTER_LANCZOS4
    ]

    def __init__(self, height, width, interp=cv2.INTER_LINEAR):
        self.op_name = self.__class__.__name__
        if not isinstance(height, int):
            raise TypeError(
                "[{}] Type of height is invalid. Must be Integer, but recieved is {}"
                .format(self.op_name, type(height)))
        if not isinstance(width, int):
            raise TypeError(
                "[{}] Type of width is invalid. Must be Integer, but recieved is {}"
                .format(self.op_name, type(width)))
        if height <= 0 or width <= 0:
            raise ValueError(
                "[{}] Height and width should be greater than 0, but recieved height/width is {}/{}".
                format(self.op_name, height, width))
        if not isinstance(interp, int):
            raise TypeError(
                "[{}] Type of interp is invalid. Must be Integer, but recieved is {}"
                .format(self.op_name, type(interp)))
        if interp not in self.interp_list:
            raise ValueError(
                "[{}] Interp should be one of {}, but recieved is {}.".format(
                    self.op_name, self.interp_list, interp))
        self.height = height
        self.width = width
        self.interp = interp

    def __call__(self, im):
        im = cv2.resize(
            im, (self.width, self.height), interpolation=self.interp)
        if im.ndim < 3:
            im = np.expand_dims(im, axis=-1)
        return im

    def infer_shape(self, shape_info):
        shape_info.shape[self.op_name] = (self.width, self.height)
        return shape_info


class CenterCrop(Transform):
    def __init__(self, height, width):
        self.op_name = self.__class__.__name__
        if not isinstance(height, int):
            raise TypeError(
                "[{}] Type of height is invalid. Must be Integer, but recieved is {}"
                .format(self.op_name, type(height)))
        if not isinstance(width, int):
            raise TypeError(
                "[{}] Type of width is invalid. Must be Integer, but recieved is {}"
                .format(self.op_name, type(width)))
        if height <= 0 or width <= 0:
            raise ValueError(
                "[{}] Height and width should be greater than 0, but recieved height/width is {}/{}".
                format(self.op_name, height, width))
        self.height = height
        self.width = width

    def __call__(self, im):
        origin_height, origin_width, _ = im.shape
        if self.height > origin_height or self.width > origin_width:
            raise Exception(
                "[{}] recieved height({}) or width({}) should not be greater than image height({}) or width({})".
                format(self.op_name, self.height, self.width, origin_height,
                       origin_width))

        w_start = (origin_width - self.width) // 2
        h_start = (origin_height - self.height) // 2
        w_end = w_start + self.width
        h_end = h_start + self.height
        im = im[h_start:h_end, w_start:w_end, :]
        return im

    def infer_shape(self, shape_info):
        shape_info.shape[self.op_name] = (self.width, self.height)
        return shape_info


class Padding(Transform):
    def __init__(self, height=0, width=0, stride=-1, im_value=[0, 0, 0]):
        self.op_name = self.__class__.__name__
        if not (isinstance(height, int) and isinstance(width, int) and
                isinstance(stride, int)):
            raise TypeError(
                "[{}] Type of height/width/stride is invalid. Must be Integer, but recieved is {}/{}/{}"
                .format(self.op_name, type(height), type(width), type(stride)))
        if height < 0 or width < 0:
            raise ValueError(
                "[{}] Height/width should be not less than 0, but recieved is {}/{}".
                format(self.op_name, height, width))
        if stride < -1:
            raise ValueError(
                "[{}] Stride should be not less than -1, but recieved is {}".
                format(self.op_name, stride))
        if not isinstance(im_value, Sequence):
            raise TypeError(
                "[{}] Type of im_value should be list or tuple, but recieved is {}".
                format(self.op_name, type(im_value)))
        self.height = height
        self.width = width
        self.stride = stride
        self.im_value = im_value

    def _compute_padded_shape(self, origin_height, origin_width):
        if isinstance(self.height, int) and isinstance(
                self.width, int) and self.height > 0 and self.width > 0:
            padded_width = self.width
            padded_height = self.height
        elif self.stride >= 1:
            padded_width = np.ceil(origin_width / self.stride * self.stride)
            padded_height = np.ceil(origin_height / self.stride * self.stride)
        else:
            raise ValueError(
                "[{}] stride(int, >=1) or height(int, >0) or width(int, >0) should be set.".
                format(self.__class__.__name__))

        pad_height = padded_height - origin_height
        pad_width = padded_width - origin_width
        if pad_height < 0 or pad_width < 0:
            raise Exception(
                'the width({}) or height({}) of input image should be less than the setting width({}) or height({})'
                .format(origin_width, origin_height, padded_width,
                        padded_height))
        pad_info = {
            'pad_height': pad_height,
            'pad_width': pad_width,
            'padded_height': padded_height,
            'padded_width': padded_width
        }
        return pad_info

    def __call__(self, im):
        origin_height, origin_width, origin_channel = im.shape
        pad_info = self._compute_padded_shape(origin_height, origin_width)
        pad_height = pad_info['pad_height']
        pad_width = pad_info['pad_width']
        padded_height = pad_info['padded_height']
        padded_width = pad_info['padded_width']
        padded_im = np.zeros(
            (padded_height, padded_width, origin_channel), dtype=np.float32)
        for i in range(origin_channel):
            padded_im[:, :, i] = np.pad(
                im[:, :, i],
                pad_width=((0, pad_height), (0, pad_width)),
                mode='constant',
                constant_values=(self.im_value[i], self.im_value[i]))

        return padded_im

    def infer_shape(self, shape_info):
        origin_width, origin_height = shape_info.get_last_shape()
        padded_height, padded_width = self._compute_padded_shape(origin_height,
                                                                 origin_width)
        shape_info.shape[self.op_name] = (padded_width, padded_height)
        return shape_op


class Clip(Transform):
    def __init__(self, min_val, max_val):
        self.op_name = self.__class__.__name__
        if not isinstance(min_val, Sequence) and isinstance(max_val, Sequence):
            raise TypeError(
                "[{}] Type of input min_val/max_val should be list or tuple, but recieved is {}/{}.".
                format(self.op_name, type(min_val), type(max_val)))

        from functools import reduce
        if reduce(lambda x, y: x * y, max_val - min_val) == 0:
            raise ValueError(
                '[{}] There should not be 0 in (max_val - min_val), but recieved is {}'.
                format(self.op_name, max_val - min_val))
        input_length_list = (len(min_val), len(max_val))
        if len(set(input_length_list)) > 1:
            raise Exception(
                "[{}] Length of min_val/max_val should be the same, but recieved is {}".
                format(self.op_name, input_length_list))
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, im):
        for k in range(im.shape[2]):
            np.clip(
                im[:, :, k], self.min_val[k], self.max_val[k], out=im[:, :, k])

        return im

    def infer_shape(self, shape_info):
        shape_info.shape[self.op_name] = shape_info.get_last_shape()
        return shape_info


class BGR2RGB(Transform):
    def __init__(self):
        self.op_name = self.__class__.__name__

    def __call__(self, im):
        im = im[:, :, ::-1]
        return im

    def infer_shape(self, shape_info):
        shape_info.shape[self.op_name] = shape_info.get_last_shape()
        return shape_info


class RGB2BGR(Transform):
    def __init__(self):
        self.op_name = self.__class__.__name__

    def __call__(self, im):
        im = im[:, :, ::-1]
        return im

    def infer_shape(self, shape_info):
        shape_info.shape[self.op_name] = shape_info.get_last_shape()
        return shape_op


class Permute(Transform):
    def __init__(self):
        self.op_name = self.__class__.__name__

    def __call__(self, im):
        im = np.swapaxes(im, 1, 2)
        im = np.swapaxes(im, 1, 0)

        return im

    def infer_shape(self, shape_info):
        shape_info.shape[self.op_name] = shape_info.get_last_shape()
        return shape_info
