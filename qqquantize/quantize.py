import torch.quantization
import copy
from qqquantize.utils import get_unique_devices_
import torch.nn as nn
from qqquantize.qconfig import DEFAULT_QAT_MODULE_MAPPING

def prepare(model, qconfig, inplace=False):
    r"""propagate qconfig and add observer"""
    if not inplace:
        model = copy.deepcopy(model)

    propagate_qconfig_(model, qconfig)

    add_observer_(model)
    return model


def propagate_qconfig_(module, qconfig):
    r"""Propagate qconfig through the module hierarchy and assign `qconfig`
    attribute on each leaf module
    """
    children = list(module.children())
    if len(children) == 0:
        module.qconfig = qconfig
    else:
        for child in children:
            propagate_qconfig_(child, qconfig)
        

def add_observer_(module):
    """Add observer for the leaf child of the module.
    This function insert observer module to all leaf child module that
    has a valid qconfig attribute.
    """

    def _observer_forward_hook(self, input, output):
        r"""Forward hook that calls observer on the output
        """
        return self.activation_post_process(output)

    devices = get_unique_devices_(module)
    assert len(devices) <= 1, (
        "add_observer_ only works with cpu or single-device CUDA modules, "
        "but got devices {}".format(devices)
    )
    device = next(iter(devices)) if len(devices) > 0 else None 
    
    for child in module.children():
        add_observer_(child)

    if hasattr(module, 'qconfig') and module.qconfig is not None and \
        len(module._modules) == 0 and not isinstance(module, torch.nn.Sequential):
        # observer and hook will be gone after we swap the module
        activation = module.qconfig.activation()
        if device is not None:
            activation.to(device)
        module.add_module('activation_post_process', activation)
        module.register_forward_hook(_observer_forward_hook)


def convert(module, mapping=None, inplace=False):
    r"""Converts the float module with observers (where we can get quantization
    parameters) to a quantized module.

    Args:
        module: calibrated module with observers
        mapping: a dictionary that maps from float module type to quantized
                 module type, can be overwrritten to allow swapping user defined
                 Modules
        inplace: carry out model transformations in-place, the original module
                 is mutated

    """
    if mapping is None:
        mapping = DEFAULT_QAT_MODULE_MAPPING
    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}

    swappable_modules = list(mapping.keys())

    for name, mod in module.named_children():
        if type(mod) not in swappable_modules:
            convert(mod, mapping, inplace=True)
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
