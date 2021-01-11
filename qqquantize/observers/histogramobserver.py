import torch
import math
from .observerbase import ObserverBase

class HistogramObserver(ObserverBase):
    r"""
    The module records the running histogram of tensor values along with
    min/max values. ``calculate_qparams`` will calculate scale and zero_point.

    Args:
        bins: Number of bins to use for the histogram
        upsample_rate: Factor by which the histograms are upsampled, this is
                       used to interpolate histograms with varying ranges across observations

    The scale and zero point are computed as follows:

    1. Create the histogram of the incoming inputs.
        The histogram is computed continuously, and the ranges per bin change
        with every new tensor observed.
    2. Search the distribution in the histogram for optimal min/max values.
        The search for the min/max values ensures the minimization of the
        quantization error with respect to the floating point model.
    3. Compute the scale and zero point the same way as in the
        :class:`~torch.quantization.MinMaxObserver`
    """

    def __init__(self, bins=2048, upsample_rate=128, bits=8):
        # bins: The number of bins used for histogram calculation.
        super().__init__()
        self.bins = bins
        self.register_buffer('histogram', torch.zeros(self.bins))
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        self.register_buffer('bits', torch.tensor([bits], dtype=torch.int))
        self.qmin = - 2 ** (bits-1)
        self.qmax = 2 ** (bits-1) - 1
        self.dst_nbins = 2 ** bits
        self.upsample_rate = upsample_rate

    @torch.jit.ignore
    def _non_linear_param_search(self):
        r"""Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        def _get_norm(delta_begin, delta_end, density, norm_type):
            r"""
            Compute the norm of the values uniformaly distributed between
            delta_begin and delta_end.

            norm = density * (integral_{begin, end} x^2)
                 = density * (end^3 - begin^3) / 3
            """
            assert norm_type == "L2", "Only L2 norms are currently supported"
            norm = 0.0
            if norm_type == "L2":
                norm = (
                    delta_end * delta_end * delta_end
                    - delta_begin * delta_begin * delta_begin
                ) / 3
            return density * norm

        def _compute_quantization_error(next_start_bin, next_end_bin, norm_type):
            r"""
            Compute the quantization error if we use start_bin to end_bin as the
            min and max to do the quantization.
            """
            bin_width = (self.max_val.item() - self.min_val.item()) / self.bins

            dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
            if dst_bin_width == 0.0:
                return 0.0

            src_bin = torch.arange(self.bins, device=self.histogram.device)
            # distances from the beginning of first dst_bin to the beginning and
            # end of src_bin
            src_bin_begin = (src_bin - next_start_bin) * bin_width
            src_bin_end = src_bin_begin + bin_width

            # which dst_bins the beginning and end of src_bin belong to?
            dst_bin_of_begin = torch.clamp(src_bin_begin // dst_bin_width, 0, self.dst_nbins - 1)
            dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

            dst_bin_of_end = torch.clamp(src_bin_end // dst_bin_width, 0, self.dst_nbins - 1)
            dst_bin_of_end_center = (dst_bin_of_end + 0.5) * dst_bin_width

            density = self.histogram / bin_width

            norm = torch.zeros(self.bins, device=self.histogram.device)

            delta_begin = src_bin_begin - dst_bin_of_begin_center
            delta_end = dst_bin_width / 2
            norm += _get_norm(delta_begin, delta_end, density, norm_type)

            norm += (dst_bin_of_end - dst_bin_of_begin - 1) * _get_norm(
                -dst_bin_width / 2, dst_bin_width / 2, density, norm_type
            )

            dst_bin_of_end_center = (
                dst_bin_of_end * dst_bin_width + dst_bin_width / 2
            )

            delta_begin = -dst_bin_width / 2
            delta_end = src_bin_end - dst_bin_of_end_center
            norm += _get_norm(delta_begin, delta_end, density, norm_type)

            return norm.sum()

        assert self.histogram.size()[0] == self.bins, "bins mistmatch"
        bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = sum(self.histogram)
        cSum = torch.cumsum(self.histogram, dim=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = _compute_quantization_error(next_start_bin, next_end_bin, "L2")

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max

    @torch.jit.ignore
    def _adjust_min_max(self, combined_min, combined_max, upsample_rate):
        # type: (Tensor, Tensor, int) -> Tuple[Tensor, Tensor, int, int]
        # We ensure that:
        # (combined_max - combined_min)/(downsample_rate*Nbins) = (max - min)/(upsample_rate*Nbins)
        # This allows us to have a common grid of resolution s, where we can align
        # the input histogram
        # start_idx maps min_val to the histogram bin index.

        hist_bin_width = (self.max_val - self.min_val) / (self.bins * upsample_rate)
        downsample_rate = torch.ceil((combined_max - combined_min) / (self.bins * hist_bin_width)).to(torch.int).item()
        e = downsample_rate * (self.bins * hist_bin_width) - (combined_max - combined_min)
        # Relax only the max, not the min, so that for one sided distributions, min stays at zero
        combined_max = combined_max + e
        combined_min = combined_min
        start_idx = torch.round((self.min_val - combined_min) / hist_bin_width).to(torch.int).item()
        return combined_min, combined_max, downsample_rate, start_idx

    @torch.jit.ignore
    def _combine_histograms(self, orig_hist, new_hist, upsample_rate, downsample_rate, start_idx, Nbins):
        # type: (Tensor, Tensor, int, int, int, int) -> Tensor
        # First up-sample the histogram with new data by a factor of L
        # This creates an approximate probability density thats piecwise constant
        upsampled_histogram = new_hist.repeat_interleave(upsample_rate)
        # Now insert the upsampled histogram into the output
        # histogram, which is initialized with zeros.
        # The offset at which the histogram is introduced is determined
        # by the start index as the output histogram can cover a wider range
        histogram_with_output_range = torch.zeros((Nbins * downsample_rate), device=orig_hist.device)
        histogram_with_output_range[start_idx:Nbins * upsample_rate + start_idx] = upsampled_histogram
        # Compute integral histogram, double precision is needed to ensure
        # that there are no overflows
        integral_histogram = torch.cumsum(histogram_with_output_range, 0,
                                          dtype=torch.double)[downsample_rate - 1 :: downsample_rate]
        # Finally perform interpolation
        shifted_integral_histogram = torch.zeros((Nbins), device=orig_hist.device)
        shifted_integral_histogram[1:Nbins] = integral_histogram[0:-1]
        interpolated_histogram = (integral_histogram - shifted_integral_histogram) / upsample_rate
        orig_hist = orig_hist + interpolated_histogram.to(torch.float)
        return orig_hist

    def forward(self, x_orig):
        # type: (Tensor) -> Tensor
        x = x_orig.detach()
        min_val = self.min_val
        max_val = self.max_val
        same_values = min_val.item() == max_val.item()
        is_uninitialized = min_val == float('inf') and max_val == float('-inf')
        if is_uninitialized or same_values:
            min_val, max_val = torch._aminmax(x)
            self.min_val.resize_(min_val.shape)
            self.min_val.copy_(min_val)
            self.max_val.resize_(max_val.shape)
            self.max_val.copy_(max_val)
            torch.histc(x, self.bins, min=min_val, max=max_val, out=self.histogram)
        else:
            new_min, new_max = torch._aminmax(x)
            combined_min = torch.min(new_min, min_val)
            combined_max = torch.max(new_max, max_val)
            # combine the existing histogram and new histogram into 1 histogram
            # We do this by first upsampling the histogram to a dense grid
            # and then downsampling the histogram efficiently
            combined_min, combined_max, downsample_rate, start_idx = \
                self._adjust_min_max(combined_min, combined_max, self.upsample_rate)
            combined_histogram = torch.histc(x, self.bins, min=combined_min, max=combined_max)
            if combined_min == min_val and combined_max == max_val:
                combined_histogram += self.histogram
            else:
                combined_histogram = self._combine_histograms(
                    combined_histogram,
                    self.histogram,
                    self.upsample_rate,
                    downsample_rate,
                    start_idx,
                    self.bins)

            self.histogram.resize_(combined_histogram.shape)
            self.histogram.copy_(combined_histogram)
            self.min_val.resize_(combined_min.shape)
            self.min_val.copy_(combined_min)
            self.max_val.resize_(combined_max.shape)
            self.max_val.copy_(combined_max)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        is_uninitialized = (self.min_val == float('inf') and
                            self.max_val == float('-inf'))
        if is_uninitialized:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0]), torch.tensor([0])
        assert self.bins == len(self.histogram), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )

        new_min, new_max = self._non_linear_param_search()

        return self._calculate_qparams(new_min, new_max)
    
    def _calculate_qparams(self, min_val, max_val):
        assert min_val.dim() == 0 and max_val.dim() == 0, 'only support per tensor'
        bits = self.bits.item()
        max_val, min_val = float(max_val), float(min_val)
        max_val = max(-min_val, max_val)
        scale = max_val / max(-self.qmin, self.qmax)
        scale = max(scale, 1e-8)
        scale = 0.5 ** round(math.log(scale, 0.5))
        zero_point = 0
        if scale == 0:
            raise ValueError('scale is 0')
        return torch.tensor([scale]), torch.tensor([zero_point])

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(HistogramObserver, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'min_val'] = self.min_val
        destination[prefix + 'max_val'] = self.max_val

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 3:
            # if min_val and max_val are not initialized, update their shape
            # to account for the differences between v2 and v3
            min_val_name, max_val_name = prefix + 'min_val', prefix + 'max_val'
            if min_val_name in state_dict:
                if state_dict[min_val_name].shape == torch.Size([0]):
                    state_dict[min_val_name] = torch.tensor(float('inf'))
            if max_val_name in state_dict:
                if state_dict[max_val_name].shape == torch.Size([0]):
                    state_dict[max_val_name] = torch.tensor(float('-inf'))

        local_state = ['min_val', 'max_val']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(HistogramObserver, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                             missing_keys, unexpected_keys, error_msgs)