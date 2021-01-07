import torch.quantization
import copy
from qqquantize.utils import get_unique_devices_
import torch.nn as nn
from qqquantize.qmodules import QStubWrapper, QStub
from qqquantize.qconfig import DEFAULT_QAT_MODULE_MAPPING

def prepare(model, qconfig, mapping=None):
    r"""propagate qconfig and convert"""    
    if mapping is None:
        mapping = DEFAULT_QAT_MODULE_MAPPING
    
    model = QStubWrapper(model)
    add_cfg_lst =  list(mapping.keys()) + [QStub]
    _propagate_qconfig(model, qconfig, add_cfg_lst)
    _convert(model, mapping, inplace=True)
    return model

def _propagate_qconfig(module, qconfig, add_cfg_list):
    r"""Propagate qconfig through the module hierarchy and assign `qconfig`
    attribute on each leaf module
    """
    children = list(module.children())
    for child in children:
        _propagate_qconfig(child, qconfig, add_cfg_list)
    add_qconfig = any([isinstance(module, valid_type) for valid_type in add_cfg_list])
    if add_qconfig:
        module.qconfig = qconfig
        
def _convert(module, mapping=None, inplace=False):
    r"""
    Args:
        module: calibrated module with observers
        mapping: a dictionary that maps from float module type to quantized
                 module type, can be overwrritten to allow swapping user defined
                 Modules
        inplace: carry out model transformations in-place, the original module
                 is mutated

    """
    assert mapping
    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}

    swappable_modules = list(mapping.keys())

    for name, mod in module.named_children():
        if type(mod) not in swappable_modules:
            _convert(mod, mapping, inplace=True)
        reassign[name] = swap_module(mod, mapping)

    for key, value in reassign.items():
        module._modules[key] = value

    return module


def swap_module(mod, mapping):
    r"""Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.

    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to nnq module

    Return:
        The corresponding quantized module of `mod`
    """
    new_mod = mod
    # Always replace dequantstub with dequantize
    if hasattr(mod, 'qconfig') and mod.qconfig is not None:
        if type(mod) in mapping:
            # respect device affinity when swapping modules
            devices = get_unique_devices_(mod)
            assert len(devices) <= 1, (
                "swap_module only works with cpu or single-device CUDA modules, "
                "but got devices {}".format(devices)
            )
            device = next(iter(devices)) if len(devices) > 0 else None
            new_mod = mapping[type(mod)].from_float(mod)
            if device:
                new_mod.to(device)
    return new_mod
