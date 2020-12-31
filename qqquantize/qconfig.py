
import torch.nn as nn
import qqquantize
from qqquantize.qmodules.qlinear import QLinear
from qqquantize.qmodules.qconv import QConv2d

DEFAULT_QAT_MODULE_MAPPING = {
    nn.Linear: QLinear,
    nn.Conv2d: QConv2d
}
