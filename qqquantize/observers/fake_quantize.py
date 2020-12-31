import torch
import torch.nn as nn
from torch.autograd.function import InplaceFunction
from .minmaxobserver import MovingAverageMinMaxObserver
from .observerbase import _with_args

class Fake_quantize_per_tensor(InplaceFunction):
    """return a quantized and dequantized a float tensor"""
    @staticmethod
    def forward(ctx, X, scale, zero_point, qmin, qmax, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(X)
        else:
            X = X.clone()

        with torch.no_grad():
            Xq = torch.floor(X / scale + zero_point)
            Xq = torch.clip(Xq, qmin, qmax)
            Xqf = (Xq - zero_point) * scale
            return Xqf

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None, None, None, None


class FakeQuantize(nn.Module):
    def __init__(self, observer=MovingAverageMinMaxObserver, **observer_kwargs):
        super().__init__()
        self.register_buffer('fake_quant_enabled', torch.tensor([0], dtype=torch.uint8))
        self.register_buffer('observer_enabled', torch.tensor([0], dtype=torch.uint8))
        self.observer = observer(**observer_kwargs)
        self.register_buffer('scale', torch.tensor([1.0]))
        self.register_buffer('zero_point', torch.tensor([0]))

    @torch.jit.export
    def enable_fake_quant(self):
        self.fake_quant_enabled[0] = 1
        return self

    @torch.jit.export
    def disable_fake_quant(self):
        self.fake_quant_enabled[0] = 0
        return self

    @torch.jit.export
    def enable_observer(self):
        self.observer_enabled[0] = 1
        return self

    @torch.jit.export
    def disable_observer(self):
        self.observer_enabled[0] = 0
        return self

    @torch.jit.export
    def calculate_qparams(self):
        return self.observer.calculate_qparams()

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            self.scale.resize_(_scale.shape)
            self.scale.copy_(_scale)
            self.zero_point.resize_(_zero_point.shape)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            X = Fake_quantize_per_tensor.apply(
                X, float(self.scale), int(self.zero_point),
                self.observer.qmin, self.observer.qmax
            )
        return X

    with_args = classmethod(_with_args)

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={},\
            scale={}, zero_point={}'.format(
            self.fake_quant_enabled, self.observer_enabled,
            self.scale, self.zero_point)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                        missing_keys, unexpected_keys, error_msgs)
