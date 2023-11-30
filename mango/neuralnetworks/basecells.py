import torch
from torch import nn
from .lazymodules import LazyConvNd, LazyBatchNormNd

DEFAULT_ACTIVATION = torch.nn.LeakyReLU()  # torch.nn.CELU()


class Squeeze(nn.Module):
    def __init__(self, from_dim: int = 0):
        super().__init__()
        self.from_dim = from_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=tuple(range(self.from_dim, x.ndim)))


class LinearCell(nn.Sequential):
    """A fully connected layer
    followed by batch norm (optional)
    and activation function (optional)
    """

    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        activation: nn.Module | None = DEFAULT_ACTIVATION,
        batch_norm: bool = True,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_params = {"device": device, "dtype": dtype}
        if in_features:
            self.append(nn.Linear(in_features, out_features, bias=bias, **factory_params))
        else:
            self.append(nn.LazyLinear(out_features, bias=bias, **factory_params))
        if batch_norm:
            self.append(nn.BatchNorm1d(out_features, **factory_params))
        if activation is not None:
            self.append(activation)


class ConvCell(nn.Sequential):
    """A convolutional connected layer
    followed by batch norm (optional)
    and activation function (optional)
    """

    def __init__(
        self,
        in_channels: int | None,
        out_channels: int,
        kernel_size: int = 3,
        activation: nn.Module | None = DEFAULT_ACTIVATION,
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
            "padding": padding if stride == 1 else kernel_size // 2,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            **factory_params,
        }
        if conv_dim and in_channels:
            if conv_dim == 1:
                conv_cls = nn.Conv1d
                bn_cls = nn.BatchNorm1d
            elif conv_dim == 2:
                conv_cls = nn.Conv2d
                bn_cls = nn.BatchNorm2d
            elif conv_dim == 3:
                conv_cls = nn.Conv3d
                bn_cls = nn.BatchNorm3d
            else:
                raise ValueError(f"conv_dim {conv_dim} not supported")

            self.append(conv_cls(in_channels, out_channels, **conv_params))
            if batch_norm:
                self.append(bn_cls(out_channels, **factory_params))
        else:
            self.append(LazyConvNd(out_channels, **conv_params))
            if batch_norm:
                self.append(LazyBatchNormNd(**factory_params))

        if activation is not None:
            self.append(activation)


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
        activation: nn.Module | None = DEFAULT_ACTIVATION,
        out_activation: bool = True,
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
        factory_params = {"device": device, "dtype": dtype}
        conv_params = {
            "out_channels": out_channels,
            "conv_dim": conv_dim,
            "padding": "same",
            "groups": groups,
            "bias": bias,
            **factory_params,
        }

        self.conv_path = nn.Sequential(
            ConvCell(
                in_channels=in_channels,
                kernel_size=kernel_size,
                activation=activation,
                batch_norm=False,
                **conv_params,
            ),
            ConvCell(
                in_channels=out_channels,
                kernel_size=kernel_size,
                activation=None,
                batch_norm=batch_norm,
                stride=stride,
                dilation=dilation,
                **conv_params,
            ),
        )
        self.residual_path = ConvCell(
            in_channels=in_channels,
            kernel_size=1,
            activation=None,
            stride=stride,
            **conv_params,
        )
        self.activation = activation if out_activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_path(x) + self.residual_path(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
