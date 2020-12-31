import torch.nn as nn
import torch.nn.functional as F

class QBatchNorm2d(nn.BatchNorm2d):
    _FLOAT_MODULE = nn.Conv2d
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, qconfig=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.activation_post_process = qconfig.activation()
        self.weight_fake_quant = qconfig.weight()

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return self.activation_post_process(
            F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.weight_fake_quant(self.running_mean) if not self.training or self.track_running_stats else None,
                self.weight_fake_quant(self.running_var) if not self.training or self.track_running_stats else None,
                self.weight_fake_quant(self.weight),
                self.weight_fake_quant(self.bias),
                bn_training, exponential_average_factor, self.eps
            )
        )