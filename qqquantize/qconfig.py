
import torch.nn as nn
import qqquantize
import qqquantize.qmodules as qm

DEFAULT_QAT_MODULE_MAPPING = {
    nn.Linear: qm.QLinear,
    nn.Conv2d: qm.QConv2d,
    nn.BatchNorm2d: qm.QBatchNorm2d,
    qm.InputStub: qm.QStub,
}
