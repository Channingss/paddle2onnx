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

import os
import os.path as osp
import numpy as np
from deploykit.common import DataBlob
import sys 

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from itertools import chain
import argparse


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def is_dynamic_shape(shape, begin_idx, end_idx):
    for index in range(begin_idx, end_idx):
        if shape[index] < 0:
            return True
    return False

class TensorRTInferConfigs(object):
    def __init__(self, ):
        self.optimize_shape_info = {}
        self.trt_logger = trt.Logger()
        
    def set_shape_info(input_name, min_shape, opt_shape, max_shape):
        if is_dynamic_shape(min_shape, 0, len(min_shape)):
            print("error")
        if is_dynamic_shape(opt_shape, 0, len(opt_shape)):
            print("error")
        if is_dynamic_shape(max_shape, 0, len(max_shape)):
            print("error")
        self.optimize_shape_info[input_name] = [min_shape, opt_shape, max_shape]

def get_input_metadata(network):
    inputs = TensorMetadata()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        inputs.add(name=tensor.name, dtype=trt.nptype(tensor.dtype), shape=tensor.shape)
    return inputs

def get_output_metadata(network):
    outputs = TensorMetadata()
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        outputs.add(name=tensor.name, dtype=trt.nptype(tensor.dtype), shape=tensor.shape)
    return outputs

class TensorRTInferenceEngine(object):
    
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    def __init__(self, model_dir, max_workspace_size, max_batch_size, trt_cache_file=None, configs=None):

        if configs is None:
            configs = TensorRTInferConfigs()

        print("TensorRT version:", trt.__version__)

        builder = trt.Builder(configs.trt_logger)
        builder.max_batch_size = max_batch_size
        build_config =  builder.create_builder_config()
        build_config.max_workspace_size = max_workspace_size

        network = builder.create_network(self.EXPLICIT_BATCH)

        self.engine = None 
        if os.path.exists(trt_cache_file):
            self.engine = self.build_engine_from_trt_file(trt_cache_file, configs.trt_logger)
        else:
            self.engine = self.build_engine_from_onnx_file(model_dir, builder, network, build_config, configs)
            with open(trt_cache_file, "wb") as f:
                f.write(self.engine.serialize())

        self.input_names = []
        self.output_names = []
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                self.input_names.append(binding)
            else:
                self.output_names.append(binding)

    def allocate_buffers(self, engine, context):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for i, binding in enumerate(engine):
            print(context.get_binding_shape(i))
            size = trt.volume(context.get_binding_shape(i)) 
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference(self, context, bindings, inputs, outputs, stream, batch_size=1):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]
    
    # This function is generalized for multiple inputs/outputs for full dimension networks.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference_v2(self, context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def build_engine_from_onnx_file(self, model_dir, builder, network, build_config, configs):
        # Takes an ONNX file and creates a TensorRT engine to run inference
        parser = trt.OnnxParser(network, configs.trt_logger) 
        if not os.path.exists(model_dir):
            print('ONNX file {} not found, t.'.format(model_dir))
            exit(0)
        with open(model_dir, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None

        # check input shape 
        profile = builder.create_optimization_profile()
        need_config_input_shape = False
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(tensor.name, tensor.shape)
            if is_dynamic_shape(tensor.shape, 1, len(tensor.shape)): 
                if tensor.name not in configs.dynamic_shape_info:
                    print("input:{} with dynamic shape {} please set configs by api:set_dynamic_shape_info.".format(tensor.name, tensor.shape))
                    need_config_input_shape = True
                else:
                    min_shape =  build_config.optimize_shape_info[tensor.name][0]
                    opt_shape =  build_config.optimize_shape_info[tensor.name][1]
                    max_shape =  build_config.optimize_shape_info[tensor.name][2]
                    profile.set_shape(tensor.name, min_shape, opt_shape, max_shape) 
            elif is_dynamic_shape(tensor.shape, 0, 1):
                rest_shape = list(tensor.shape[1:])
                min_shape = [1] + rest_shape
                opt_batch = [max_batch_size // 2] if builder.max_batch_size > 2  else [1]
                opt_shape = opt_batch + rest_shape
                max_shape = [builder.max_batch_size] + rest_shape
                profile.set_shape(tensor.name, min_shape , opt_shape, max_shape) 
            else:
                print(1111)


        if need_config_input_shape:
            exit(0)
        build_config.add_optimization_profile(profile)

        engine = builder.build_engine(network, build_config)
        print('Completed build engine of ONNX file')
        return engine

    def build_engine_from_trt_file(self, trt_cache_file, trt_logger):
        # If a serialized trt engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(trt_cache_file))
        with open(trt_cache_file, "rb") as f, trt.Runtime(trt_logger) as runtime:
            return  runtime.deserialize_cuda_engine(f.read())

    def infer(self, input_blobs):
        # Do inference
        context = self.engine.create_execution_context()

        #input_blobs[0].data = np.concatenate([input_blobs[0].data,input_blobs[0].data], axis=0)
        input_blobs[0].data = input_blobs[0].data
        for i, binding_name in enumerate(self.engine):
            if self.engine.binding_is_input(binding_name):
                binding_index = self.engine.get_binding_index(binding_name)
                context.set_binding_shape(self.engine[binding_name], input_blobs[i].data.shape)

        assert context.all_binding_shapes_specified

        inputs, outputs, bindings, stream = self.allocate_buffers(self.engine, context)

        for i in range(len(inputs)):
            data = input_blobs[i].data.ravel()
            np.copyto(inputs[i].host, data)

        trt_outputs = self.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        output_blobs = []
        for index, binding_name in enumerate(self.output_names):
            binding_index = self.engine.get_binding_index(binding_name)
            output_shape = context.get_binding_shape(binding_index) 
            output_blob = DataBlob()
            output_blob.name = binding_name 
            output_blob.data =  trt_outputs[index].reshape(output_shape)
            print(output_blob.data)
            #output_data_blob.lod = output_tensor.lod()
            output_blobs.append(output_blob)

        return output_blobs
