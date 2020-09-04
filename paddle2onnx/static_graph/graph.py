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
from paddle.fluid.executor import scope_guard
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
                 output_spec=None,
                 prune_iout=False):
        self.program = None 
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.model = model
        self.to_static_program(prune_iout)

    def get_inout_spec(self, all_vars, target_vars, return_name=False):
        result_list = []
        valid_var_dict = {}
        valid_vars = [var for var in all_vars if isinstance(var, Variable)]
        for var in valid_vars:
            valid_var_dict[var.name] = var
        if target_vars is not None:
            for i, var in enumerate(target_vars):
                # check target var whether exists
                if var.name not in valid_var_dict:
                    raise RuntimeError(
                        "The variable to feed/fetch are not exist.")
                result_list.append(valid_var_dict[var.name])
        else:
            result_list = valid_vars
        if return_name:
            return result_list,  [var.name for var in result_list]
        return result_list

    def prune_input_output(self, main_program):
        feeded_vars, feeded_var_names= self.get_inout_spec(self.program.inputs, self.input_spec, True)
        target_vars = self.get_inout_spec(self.program.outputs, self.output_spec)
        main_program = main_program.clone()
        global_block = main_program.global_block()
        need_to_remove_op_index = []
        for i, op in enumerate(global_block.ops):
            op.desc.set_is_target(False)
            if op.type == "feed" or op.type == "fetch":
                need_to_remove_op_index.append(i)

        for index in need_to_remove_op_index[::-1]:
            global_block._remove_op(index)

        main_program.desc.flush()

        main_program = main_program._prune_with_input(
            feeded_var_names=feeded_var_names, targets=target_vars)
        main_program = main_program._inference_optimize(prune_read_op=True)
        fetch_var_names = [v.name for v in target_vars]

        prepend_feed_ops(main_program, feeded_var_names)
        append_fetch_ops(main_program, fetch_var_names)

        self.program.outputs = tuple(target_vars) 
        self.program.inputs = tuple(feeded_vars) 
        return main_program

    @switch_to_static_graph
    def to_static_program(self, prune_iout=False):
        paddle.jit.set_verbosity(10)
        if isinstance(self.model, TranslatedLayer):
            # TODO 待提供concrete_program的接口
            concrete_program  =  self.model.program()
        elif isinstance(self.model, Layer):
            if isinstance(self.model.forward, StaticLayer):
                #concrete_program, _ = self.model.forward.get_concrete_program(*self.input_spec)
                concrete_program  = self.model.forward.concrete_program
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
        if prune_iout: 
            self.program.main_program = self.prune_input_output(self.program.main_program.clone())

