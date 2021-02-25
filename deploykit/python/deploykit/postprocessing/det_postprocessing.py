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
# limitations under the License.

from deploykit.common import ConfigParser


class Coordinate(object):
    def __init__(self):
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None
        self.width = None
        self.height = None


class Mask(object):
    def __init__(self):
        self.binary_mask = None


class Bbox(object):
    def __init__(self):
        self.coordinate = Coordinate()
        self.mask = Mask()
        self.category_id = None
        self.category = None
        self.score = None


class DetResult(object):
    def __init__(self):
        self.bboxes = list()


def expand_boxes(boxes, scale):
    """
    Expand an array of boxes by a given scale.
    """
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


class DetPostprocessor(object):
    def __init__(self, config):
        if not isinstance(config, ConfigParser):
            raise TypeError(
                "Type of config must be ConfigParser, but recieved is {}".
                format(type(config)))
        self.architecture = config.config['model_name']
        self.labels = config.config['labels']
        self.catid2label = dict(
            {i: self.labels[i]
             for i in range(len(self.labels))})
        if 'mask_resolution' in config.config and config.config[
                'mask_resolution'] > 0:
            import pycocotools.mask as mask_util
            self.mask_resolution = config.config['mask_resolution']

    def offset_to_lengths(self, lod):
        offset = lod[0]
        lengths = [offset[i + 1] - offset[i] for i in range(len(offset) - 1)]
        return lengths

    def get_binary_mask(self, mask, bbox, im_shape, thresh_binarize=0.5):
        xmin, ymin, xmax, ymax = bbox
        origin_width, origin_height = im_shape
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        w = np.maximum(w, 1)
        h = np.maximum(h, 1)

        resized_mask = cv2.resize(mask, (w, h))
        resized_mask = np.array(resized_mask > thresh_binarize, dtype=np.uint8)
        im_mask = np.zeros((origin_width, origin_height), dtype=np.uint8)

        x0 = min(max(xmin, 0), origin_width)
        x1 = min(max(xmax + 1, 0), origin_height)
        y0 = min(max(ymin, 0), origin_height)
        y1 = min(max(ymax + 1, 0), origin_height)

        im_mask[y0:y1, x0:x1] = resized_mask[(y0 - ymin):(y1 - ymin), (
            x0 - xmin):(x1 - xmin)]
        segm = mask_util.encode(
            np.array(
                im_mask[:, :, np.newaxis], order='F'))[0]
        segm['counts'] = segm['counts'].decode('utf8')
        return segm

    def postprocess(self,
                    bbox_blob,
                    shape_info,
                    mask_blob=None,
                    thresh_binaraize=0.5):
        det_results = (DetResult(), ) * len(shape_info)
        from functools import reduce
        if reduce(lambda x, y: x * y,
                  bbox_blob.data.shape) < 6 or bbox_blob.data is None:
            return det_results

        lengths = self.offset_to_lengths(bbox_blob.lod)
        start_bbox_id = 0
        for im_id in range(len(lengths)):
            bbox_num = lengths[im_id]
            bbox = bbox_blob.data[start_bbox_id:start_bbox_id + bbox_num][:, 2:]
            catid_scores = bbox_blob.data[start_bbox_id:start_bbox_id +
                                          bbox_num][:, 0:2]
            if mask_blob is not None:
                mask = mask_blob.data[start_bbox_id:start_bbox_id + bbox_num]
                scale = (self.mask_resolution + 2.0) / self.mask_resolution
                expand_bbox = expand_boxes(bbox, scale).astype(np.int32)
                padded_mask = np.zeros(
                    (self.mask_resolution + 2, self.mask_resolution + 2),
                    dtype=np.float32)
            start_bbox_id += bbox_num

            for i in range(bbox_num):
                xmin, ymin, xmax, ymax = bbox[i]
                catid, score = catid_scores[i]
                width = xmax - xmin + 1
                height = ymax - ymin + 1

                bbox_object = Bbox()
                bbox_object.coordinate.xmin = xmin
                bbox_object.coordinate.ymin = ymin
                bbox_object.coordinate.xmax = xmax
                bbox_object.coordinate.ymax = ymax
                bbox_object.coordinate.width = width
                bbox_object.coordinate.height = height
                bbox_object.category_id = catid
                bbox_object.category = self.catid2label[catid]
                bbox_object.score = score

                if mask_blob is not None:
                    padded_mask[1:-1, 1:-1] = mask[i, catid, :, :]
                    bbox_object.mask = self.get_binary_mask(
                        padded_mask, expand_bbox[i],
                        shape_info[im_id]['Origin'])

                det_results[im_id].bboxes.append(bbox_object)
        return det_results

    def __call__(self, outputs, shape_info_list):
        bbox_blob = outputs[0]
        mask_blob = None
        if hasattr(self, 'mask_resolution'):
            mask_blob = outputs[1]
        det_results = self.postprocess(bbox_blob, shape_info_list, mask_blob)
        return det_results
