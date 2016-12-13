"""Dropout variants for multithreaded deep network training"""

import lasagne
import theano.tensor as T

class DropoutLayerOverlapping(lasagne.layers.DropoutLayer):
    """Dropout layer which may have overlaps between worker threads"""

    def __init__(self, incoming, mask=None, **kwargs):
        super(DropoutLayerOverlapping, self).__init__(incoming, **kwargs)
        self.mask = mask

    def get_output_for(self, input, deterministic=False, **kwargs):
        # pylint: disable=redefined-builtin,unused-argument
        if deterministic or self.p == 0:
            return input
        elif self.mask:
            return input * self.mask
        else:
            # Using theano constant to prevent upcasting
            one = T.constant(1)

            retain_prob = one - self.p
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            mask_shape = self.input_shape
            if any(s is None for s in mask_shape):
                mask_shape = input.shape

            # apply dropout, respecting shared axes
            if self.shared_axes:
                shared_axes = tuple(a if a >= 0 else a + input.ndim
                                    for a in self.shared_axes)
                mask_shape = tuple(1 if a in shared_axes else s
                                   for a, s in enumerate(mask_shape))
            mask = self._srng.binomial(mask_shape, p=retain_prob,
                                       dtype=input.dtype)
            if self.shared_axes:
                bcast = tuple(bool(s == 1) for s in mask_shape)
                mask = T.patternbroadcast(mask, bcast)
            return input * mask
