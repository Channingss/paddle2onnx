import os
import pickle
import warnings
import inspect

import six
from collections import OrderedDict
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

def prepend_feed_ops(inference_program,
                     feed_target_names,
                     feed_holder_name='feed'):
    if len(feed_target_names) == 0:
        return

    global_block = inference_program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True)

    for i, name in enumerate(feed_target_names):
        if not global_block.has_var(name):
            raise ValueError(
                "The feeded_var_names[{i}]: '{name}' doesn't exist in pruned inference program. "
                "Please check whether '{name}' is a valid feed_var name, or remove it from feeded_var_names "
                "if '{name}' is not involved in the target_vars calculation.".
                format(
                    i=i, name=name))
        out = global_block.var(name)
        global_block._prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [out]},
            attrs={'col': i})


def append_fetch_ops(inference_program,
                     fetch_target_names,
                     fetch_holder_name='fetch'):
    global_block = inference_program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True)

    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i})

class StaticGraph():
    def __init__(self,
                 model,
                 input_spec,
                 output_spec=None):
        self.program = None
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.model = model
        self.to_static_program()

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

    def generate_input_spec_by_name(self, all_vars, target_vars, return_name=False):
        result_list = []
        valid_var_dict = {}
        valid_vars = [var for var in all_vars if isinstance(var, Variable)]
        for var in valid_vars:
            valid_var_dict[var.name] = var
        if target_vars:
            for i, var in enumerate(target_vars):
                # check target var whether exists
                if var is None: 
                    continue
                if var.name not in valid_var_dict:
                    raise RuntimeError(
                        "The variable to feed/fetch are not exist.")
                result_list.append(valid_var_dict[var.name])
        else:
            result_list = valid_vars
        if return_name:
            result_list = [var.name for var in result_list]

        return result_list 

    def generate_output_spec_by_order(self, all_vars, target_vars):
        result_list = []
        if isinstance(target_vars, dict):
            order_vars = OrderedDict(target_vars)
            for i, name in order_vars.items():
                result_list.append(all_vars[i])
        if isinstance(target_vars, list):
            for i in target_vars:
                result_list.append(all_vars[i])
        else:
            valid_vars = [var for var in all_vars if isinstance(var, Variable)]
            result_list = valid_vars
        return result_list 

    def prune_input_output(self):
        input_var_names = self.generate_input_spec_by_name(self.program.inputs, self.input_spec, True)
        output_vars = self.generate_output_spec_by_order(self.program.outputs, self.output_spec)
        self.program.main_program = self.program.main_program._prune_with_input(
        feeded_var_names=input_var_names, targets=output_vars)
        self.program.main_program = self.program.main_program._inference_optimize(prune_read_op=True)
        output_var_names = [v.name for v in output_vars]
        prepend_feed_ops(self.program.main_program, input_var_names)
        append_fetch_ops(self.program.main_program, output_var_names) 
        self.program.outputs = tuple(output_vars) 

    @switch_to_static_graph
    def to_static_program(self):
        paddle.jit.set_verbosity(10)
        if isinstance(self.model, TranslatedLayer):
            # TODO 待提供concrete_program的接口
            concrete_program  =  self.model._program_holder_dict['forward'].infer_program
        elif isinstance(self.model, Layer):
            if isinstance(self.model.forward, StaticLayer):
                concrete_program, _ = self.model.forward.get_concrete_program(*self.input_spec)
            elif inspect.ismethod(self.model.forward):
                concrete_program, _ = declarative(self.model.forward).get_concrete_program(*self.input_spec)
            else: 
                raise TypeError(
                    "The foward of model should be 'Function' or 'StaticLayer', but received layer type is %s."
                    % type(self.model))
        elif isinstance(self.model, StaticLayer):
            concrete_program, _ = self.model.get_concrete_program(*self.input_spec)
        else:
            raise TypeError(
                "The input model should be 'Layer','StaticLayer' or 'TranslatedLayer', but received layer type is %s."
                % type(self.model))
        self.program = concrete_program
        self.prune_input_output()
