

def enable_fake_quant(module):
    for mod in module.modules():
        if hasattr(mod, 'enable_fake_quant'):
            mod.enable_fake_quant()

def disable_fake_quant(module):
    for mod in module.modules():
        if hasattr(mod, 'disable_fake_quant'):
            mod.disable_fake_quant()

def enable_observer(module):
    for mod in module.modules():
        if hasattr(mod, 'enable_observer'):
            mod.enable_observer()

def disable_observer(module):
    for mod in module.modules():
        if hasattr(mod, 'disable_observer'):
            mod.disable_observer()

def get_unique_devices_(module):
    return {p.device for p in module.parameters()} | \
        {p.device for p in module.buffers()}
