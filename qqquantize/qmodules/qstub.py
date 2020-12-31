from torch import nn

__all__ = ['QStub', 'QStubWrapper']

class QStub(nn.Module):
    r"""Quantize stub module, before calibration, this is same as an observer,
    it will be swapped as `nnq.Quantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """
    def __init__(self, qconfig=None):
        super().__init__()
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x):
        return x

class QStubWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        qconfig = module.qconfig if hasattr(module, 'qconfig') else None
        self.add_module('quant', QStub(qconfig))
        self.add_module('module', module)
        self.train(module.training)

    def forward(self, X):
        X = self.quant(X)
        return self.module(X)
