import torch
from qqquantize.observers.fake_quantize import FakeQuantize
from qqquantize.qconfig import DEFAULT_QAT_MODULE_MAPPING
import qqquantize.qmodules as qm
import math

def get_weight_act_bit_dict(model, qmodules=None):
    bit_dict = {}
    if qmodules is None:
        qmodules = list(DEFAULT_QAT_MODULE_MAPPING.values())
    for mod_name, mod in model.named_modules():
        if type(mod) in qmodules:
            bit_dict[mod_name] = {}
            for m_name, m in mod.named_modules():
                # m_name = m_name.split('.')[-1]
                if isinstance(m, FakeQuantize):
                    fbit = -math.log2(m.scale.item())
                    bit_dict[mod_name][m_name] = fbit
    return bit_dict

"""
To find out tensor's source layer
assign each output a fxname, and save the input tensor's fxname
"""
class GetInpSourceHook:
    def __init__(self, mod_name):
        self.mod_name = mod_name
        self.src_names = []
    def __call__(self, module, inputs, outputs):
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs, )
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    out.fxname = f'{self.mod_name}:{i}'
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor) and hasattr(inp, 'fxname'):
                self.src_names.append(inp.fxname)

def get_input_source_dict(model, fake_input, qmodules=None):
    source_dict = {}
    hooks = []
    if qmodules is None:
        qmodules = list(DEFAULT_QAT_MODULE_MAPPING.values())
    for mod_name, mod in model.named_modules():
        if type(mod) in qmodules:
            h = GetInpSourceHook(mod_name)
            hooks.append(h)
            mod.register_forward_hook(h)
    _ = model(fake_input)
    for mod_name, mod in model.named_modules():
        for h_name, h in list(mod._forward_hooks.items()):
            if isinstance(h, GetInpSourceHook):
                source_dict[mod_name] = h.src_names
            mod._forward_hooks.pop(h_name)
    return source_dict

"""
calling get_input_source_dict and get_weight_act_bit_dict
and merge data
"""
def get_all_bits(model, fake_input, qmodules=None):
    bits = get_weight_act_bit_dict(model)
    inp_src = get_input_source_dict(model, fake_input)
    for mod_name, src_names in inp_src.items():
        for src in src_names:
            pass

"""
Get each layer's input weight and act float bits
"""
class GetBitsHook:
    def __init__(self, mod_name):
        self.mod_name = mod_name
        self.bits = {}
    def __call__(self, module, inputs, outputs):
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs, )
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor) and hasattr(out, 'scale'):
                    b = -math.log2(out.scale)
                    self.bits.setdefault('out', []).append(b)
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor) and hasattr(inp, 'scale'):
                b = -math.log2(out.scale)
                self.bits.setdefault('inp', []).append(b)
        for name, module in module.named_children():
            if isinstance(module, FakeQuantize):
                b = -math.log2(module.scale)
                self.bits[name] = b

def get_layer_bits(model, fake_input, qmodules=None):
    bits_dict = {}
    if qmodules is None:
        qmodules = list(DEFAULT_QAT_MODULE_MAPPING.values())

    for mod_name, mod in model.named_modules():
        if type(mod) in qmodules:
            mod.register_forward_hook(GetBitsHook(mod_name))
    
    _ = model(fake_input)

    for mod_name, mod in model.named_modules():
        for h_name, h in list(mod._forward_hooks.items()):
            if isinstance(h, GetBitsHook):
                bits_dict[mod_name] = h.bits
            mod._forward_hooks.pop(h_name)
    return bits_dict

def conv_bit_adjust(module, i_bit):
    assert isinstance(module, qm.QConv2d)
    device = module.act_quant.scale.device
    w_bit = -math.log2(module.weight_quant.scale)
    a_bit = -math.log2(module.act_quant.scale)
    new_a_bit = a_bit
    if i_bit + w_bit - a_bit < 0:
        new_a_bit = i_bit + w_bit
    if i_bit + w_bit - a_bit > 7:
        new_a_bit = i_bit + w_bit - 7
    module.act_quant.scale = torch.tensor([2**-new_a_bit], device=device)
    if new_a_bit != a_bit:
        return True
    else:
        return False



