import numpy as np
import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_models.layers import conv2d, CausalConv1D


class TestConv2d:
    def test_same_shape_no_dilation(self):
        x = torch.randn(1, 1, 5, 5)
        conv = conv2d(1, 1, 3)
        with torch.no_grad():
            out = conv(x)
        assert out.shape[2:] == x.shape[2:]

    def test_same_shape_with_dilation(self):
        x = torch.randn(1, 1, 5, 5)
        conv = conv2d(1, 1, 3, dilation=2)
        with torch.no_grad():
            out = conv(x)
        assert out.shape[2:] == x.shape[2:]


class TestCausalConv1d:
    def test_same_shape_no_dilation(self):
        x = torch.randn(1, 1, 6)
        conv1d = CausalConv1D(1, 1, 3)
        with torch.no_grad():
            out = conv1d(x)
        assert out.shape[2:] == x.shape[2:]

    def test_same_shape_with_dilation(self):
        x = torch.randn(1, 1, 6)
        conv1d = CausalConv1D(1, 1, 3, dilation=2)
        with torch.no_grad():
            out = conv1d(x)
        assert out.shape[2:] == x.shape[2:]

    def test_causality_no_dilation(self):
        stride = 1
        length = 6
        dilation = 1
        kernel_size = 3
        x = torch.randn(1, 1, length)
        conv1d = CausalConv1D(1, 1, kernel_size, stride, dilation, bias=False)
        with torch.no_grad():
            actual = conv1d(x)
        actual = actual.numpy().squeeze()
        weight = conv1d.weight.detach().clone().squeeze().numpy()
        padding = (int((kernel_size - 1) * dilation), 0)
        padded_x = F.pad(x, padding).detach().squeeze().numpy()
        expected = []
        for i in range(length):
            expected.append(weight @ padded_x[i : i + 3])
        expected = np.asarray(expected)
        assert np.allclose(actual, expected)
