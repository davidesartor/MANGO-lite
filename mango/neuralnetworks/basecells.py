from typing import Any, Sequence
import torch
from torch import nn
from .lazymodules import LazyConvNd, LazyBatchNormNd


class Squeeze(nn.Module):
    def __init__(self, from_dim: int = 0):
        super().__init__()
        self.from_dim = from_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=tuple(range(self.from_dim, x.ndim)))


class LinearCell(nn.Sequential):
    """A fully connected layer
    preeceded by batch norm (optional)
    and followed by activation function (optional)
    """

    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        activation: nn.Module | None = nn.ReLU(),
        batch_norm: bool = True,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_params = {"device": device, "dtype": dtype}

        if in_features:
            self.append(nn.Linear(in_features, out_features, bias, **factory_params))
            if batch_norm:
                self.append(nn.BatchNorm1d(in_features, **factory_params))
            if activation is not None:
                self.append(activation)
        else:
            self.append(nn.LazyLinear(out_features, bias, **factory_params))
            if batch_norm:
                self.append(nn.LazyBatchNorm1d(**factory_params))
            if activation is not None:
                self.append(activation)


class ConvCell(nn.Sequential):
    """A convolutional connected layer
    preeceded by batch norm (optional)
    and followed by activation function (optional)
    """

    def __init__(
        self,
        in_channels: int | None,
        out_channels: int,
        kernel_size: int = 3,
        activation: nn.Module | None = nn.ReLU(),
        batch_norm: bool = True,
        conv_dim: int | None = None,
        stride: int = 1,
        padding: int | str = "same",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_params = {"device": device, "dtype": dtype}
        conv_params = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
        }

        if not conv_dim or not in_channels:
            self.append(LazyConvNd(out_channels, **conv_params, **factory_params))
            if batch_norm:
                self.append(LazyBatchNormNd(**factory_params))
            if activation is not None:
                self.append(activation)

        elif conv_dim == 1:
            self.append(nn.Conv1d(in_channels, out_channels, **conv_params, **factory_params))
            if batch_norm:
                self.append(nn.BatchNorm1d(in_channels, **factory_params))
            if activation is not None:
                self.append(activation)

        elif conv_dim == 2:
            self.append(nn.Conv2d(in_channels, out_channels, **conv_params, **factory_params))
            if batch_norm:
                self.append(nn.BatchNorm2d(in_channels, **factory_params))
            if activation is not None:
                self.append(activation)

        elif conv_dim == 3:
            self.append(nn.Conv3d(in_channels, out_channels, **conv_params, **factory_params))
            if batch_norm:
                self.append(nn.BatchNorm3d(in_channels, **factory_params))
            if activation is not None:
                self.append(activation)
        else:
            raise ValueError(f"conv_dim {conv_dim} not supported")


class ResConvCell(nn.Module):
    """A residual convolutional connected layer
    preeceded by batch norm (optional)
    and followed by activation function (optional)
    """

    def __init__(
        self,
        in_channels: int | None,
        out_channels: int,
        kernel_size: int = 3,
        activation: nn.Module | None = nn.ReLU(),
        batch_norm: bool = True,
        conv_dim: int | None = None,
        stride: int = 1,
        padding: str | int = "same",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.conv_path = nn.Sequential(
            ConvCell(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                batch_norm=batch_norm,
                conv_dim=conv_dim,
                stride=stride,
                padding="same",
                dilation=dilation,
                groups=groups,
                bias=bias,
                device=device,
                dtype=dtype,
            ),
            ConvCell(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation=None,
                batch_norm=batch_norm,
                conv_dim=conv_dim,
                stride=stride,
                padding="same",
                dilation=dilation,
                groups=groups,
                bias=bias,
                device=device,
                dtype=dtype,
            ),
        )
        self.residual_path = ConvCell(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=None,
            batch_norm=batch_norm,
            conv_dim=conv_dim,
            stride=stride,
            padding="same",
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_path(x) + self.residual_path(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
