from typing import Any, Optional

import torch
from torch.nn.modules import conv, batchnorm


class _LazyParamDepMixin:
    def set_class_behaviour(self, param: Any):
        """set the behaviour of the preinitialized module inferred from the input"""
        ...

    def set_class_name(self, param: Any):
        """set the class to the correct one inferred at runtime"""
        ...

    # intercept the call to and setup the module behaviour based on the input
    def initialize_parameters(self, input: torch.Tensor) -> None:
        if self.has_uninitialized_params():  # type: ignore
            self.set_class_behaviour(input)
            super().initialize_parameters(input)  # type: ignore

    def _infer_parameters(self, module, input) -> None:
        super()._infer_parameters(module, input)  # type: ignore[misc]
        self.set_class_name(*input)


class LazyConvNd(_LazyParamDepMixin, conv.LazyConv1d):
    cls_to_become = None

    # we assume input is batched so we can ovveride this
    # to cut of the _get_num_spatial_dims call entirely
    def _get_in_channels(self, input: torch.Tensor) -> int:
        return input.shape[1]

    # need to override this so that foward can modify the conv_func at runtime
    def _conv_forward(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.conv_func(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def set_class_name(self, input: torch.Tensor) -> None:
        conv_dim = input.ndim - 2
        if conv_dim == 1:
            self.__class__ = conv.Conv1d
        elif conv_dim == 2:
            self.__class__ = conv.Conv2d
        elif conv_dim == 3:
            self.__class__ = conv.Conv3d

    def set_class_behaviour(self, input: torch.Tensor) -> None:
        conv_dim = input.ndim - 2
        if conv_dim == 1:
            self._conv_func = torch.nn.functional.conv1d
        elif conv_dim == 2:
            self.conv_func = torch.nn.functional.conv2d
        elif conv_dim == 3:
            self.conv_func = torch.nn.functional.conv3d
        else:
            raise ValueError(f"input shape {input.shape} incompatible with conv")

        # need to set the correct dims for kernel_size, stride, padding, dilation
        self.kernel_size = self.kernel_size * conv_dim
        self.stride = self.stride * conv_dim
        if not isinstance(self.padding, str):
            self.padding = self.padding * conv_dim
        self.dilation = self.dilation * conv_dim


class LazyBatchNormNd(_LazyParamDepMixin, batchnorm.LazyBatchNorm1d):
    cls_to_become = None

    def set_class_name(self, input: torch.Tensor) -> None:
        extra_dim = input.ndim - 2
        if extra_dim == 1 or extra_dim == 0:
            self.__class__ = batchnorm.BatchNorm1d
        elif extra_dim == 2:
            self.__class__ = batchnorm.BatchNorm2d
        elif extra_dim == 3:
            self.__class__ = batchnorm.BatchNorm3d
        else:
            raise ValueError(f"input shape {input.shape} incompatible with batchnorm")

    def set_class_behaviour(self, input: torch.Tensor) -> None:
        self.extra_dim = input.ndim - 2

    def _check_input_dim(self, input):
        if self.extra_dim == 1:
            super()._check_input_dim(input)
        elif input.dim() - 2 != self.extra_dim:
            raise ValueError(f"expected {self.extra_dim}D input, got shape:{input.shape}")
