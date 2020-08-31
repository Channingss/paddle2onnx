import os
import pickle
import warnings

import six
from paddle.fluid import core
from paddle.fluid.compiler import BuildStrategy, CompiledProgram, ExecutionStrategy
from paddle.fluid.data_feeder import check_type
from paddle.fluid.dygraph.base import program_desc_tracing_guard, switch_to_static_graph
from paddle.fluid.dygraph.dygraph_to_static.error import ERROR_DATA
from paddle.fluid.dygraph.dygraph_to_static.program_translator import FunctionSpec, ProgramTranslator
from paddle.fluid.dygraph.io import EXTRA_VAR_INFO_FILENAME, VARIABLE_FILENAME, TranslatedLayer
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid.executor import Executor, scope_guard
from paddle.fluid.framework import Block, ParamBase, Program, Variable
from paddle.fluid.framework import _current_expected_place, _dygraph_guard, _dygraph_tracer
from paddle.fluid.framework import dygraph_only, in_dygraph_mode
from paddle.fluid.wrapped_decorator import wrap_decorator

class StaticGraph():
    def __init__(self,
                 model,
                 input_spec):
        self.program = self.to_static_program(model, input_spec=input_spec)

    @switch_to_static_graph
    def to_static_program(self, layer, input_spec=None, configs=None):
        # 1. input check
        prog_translator = ProgramTranslator()
        if not prog_translator.enable:
            raise RuntimeError(
                "static graph doesn't work when setting ProgramTranslator.enable=False."
            )
        if not isinstance(layer, Layer):
            raise TypeError(
                "The input layer should be 'Layer', but received layer type is %s."
                % type(layer))
    
        if input_spec is not None:
            if not isinstance(input_spec, list):
                raise TypeError(
                    "The input input_spec should be 'list', but received input_spec's type is %s."
                    % type(input_spec))
            for var in input_spec:
                if not isinstance(var, core.VarBase):
                    raise TypeError(
                        "The element in input_spec list should be 'Variable', but received element's type is %s."
                        % type(var))
    
        # 2. get program of declarative Layer.forward
        prog_cache = prog_translator.get_program_cache()
        # make dummy args & kwargs, to get excepted FunctionSpec
        layer_func = FunctionSpec(type(layer).forward, [layer], {})
        concrete_program, _ = prog_cache.get_program(layer_func)
    
        return concrete_program 
