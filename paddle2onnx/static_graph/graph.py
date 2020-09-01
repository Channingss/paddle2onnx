import os
import pickle
import warnings
import inspect

import six
import paddle
from paddle.fluid import core
from paddle.fluid.dygraph.base import program_desc_tracing_guard, switch_to_static_graph
from paddle.fluid.dygraph.dygraph_to_static.program_translator import FunctionSpec, ProgramTranslator, StaticLayer
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid.framework import Block, ParamBase, Program, Variable
from paddle.fluid.framework import dygraph_only, in_dygraph_mode
from paddle.fluid.dygraph.io import TranslatedLayer
from paddle.static import InputSpec
from paddle.fluid.dygraph import declarative

class StaticGraph():
    def __init__(self,
                 model,
                 input_spec):
        self.program = self.to_static_program(model, input_spec=input_spec)

    def _verify_input_spec(self, input_spec):
        """
        Verifies the `input_spec` and its element type is valid.
        """
        if not isinstance(input_spec, (tuple, list)):
            raise TypeError(
                "The type(input_spec) should be one of (tuple, list), but received {}.".
                format(type_name(input_spec)))
        input_spec = tuple(input_spec)
        for spec in flatten(input_spec):
            if not isinstance(spec, paddle.static.InputSpec):
                raise ValueError(
                    "The type(elem) from input_spec should be `InputSpec`, but received {}.".
                    format(type_name(spec)))

        return input_spec

    @switch_to_static_graph
    def to_static_program(self, layer, input_spec=None, configs=None):
        paddle.jit.set_verbosity(10)
        if isinstance(layer, TranslatedLayer):
            # TODO 待提供concrete_program的接口
            concrete_program  =  layer._program_holder_dict['forward'].infer_program
        elif isinstance(layer, Layer):
            if isinstance(layer.forward, StaticLayer):
                concrete_program, _ = layer.forward.get_concrete_program(input_spec)
            elif inspect.ismethod(layer.forward):
                concrete_program, _ = declarative(layer.forward).get_concrete_program(input_spec)
            else: 
                raise TypeError(
                    "The foward of model should be 'Function' or 'StaticLayer', but received layer type is %s."
                    % type(layer))
        elif isinstance(layer, StaticLayer):
            concrete_program, _ = layer.get_concrete_program(input_spec)
        else:
            raise TypeError(
                "The input model should be 'Layer','StaticLayer' or 'TranslatedLayer', but received layer type is %s."
                % type(layer))
            
        return concrete_program 
