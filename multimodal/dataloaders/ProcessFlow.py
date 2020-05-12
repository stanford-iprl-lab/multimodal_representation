import torch
import numpy as np


class ProcessFlow(object):
    """Process optical flow into a pyramid.
    Args:
        pyramid_scale (list): scaling factors to downsample
            the spatial pyramid
    """

    def __init__(self, pyramid_scales=[2, 4, 8]):
        assert isinstance(pyramid_scales, list)
        self.pyramid_scales = pyramid_scales

    def __call__(self, sample):
        # subsampling to create small flow images
        for scale in self.pyramid_scales:
            scaled_flow = sample['flow'][::scale, ::scale]
            sample['flow{}'.format(scale)] = scaled_flow
        return sample
