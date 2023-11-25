from typing import Sequence
import torch
from . import basecells

DEFAULT_ACTIVATION = torch.nn.CELU()


def squash(net: torch.nn.Sequential) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        *[squash(layer) if isinstance(layer, torch.nn.Sequential) else layer for layer in net]
    )


class LinearNet(torch.nn.Sequential):
    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        hidden_features: Sequence[int] = (16,),
        activation: torch.nn.Module = DEFAULT_ACTIVATION,
        out_activation: torch.nn.Module | None = None,
        batch_norm: bool = True,
        out_batch_norm: bool = True,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_params = {"device": device, "dtype": dtype}
        layer_sizes = [in_features, *hidden_features, out_features]
        activations = [activation for _ in hidden_features] + [out_activation]
        batch_norms = [batch_norm for _ in hidden_features] + [out_batch_norm]
        for in_size, out_size, act, bn in zip(
            layer_sizes[:-1], layer_sizes[1:], activations, batch_norms
        ):
            cell = basecells.LinearCell(
                in_features=in_size,
                out_features=out_size,
                activation=act,
                batch_norm=bn,
                bias=bias,
                **factory_params,
            )
            self.append(cell)


class ConvNet(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int | None,
        out_channels: int,
        kernel_size: int = 3,
        hidden_channels: Sequence[int] = (16,),
        activation: torch.nn.Module = DEFAULT_ACTIVATION,
        out_activation: torch.nn.Module | None = None,
        batch_norm: bool = True,
        out_batch_norm: bool = True,
        residual_connections: bool = False,
        groups: int = 1,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_params = {"device": device, "dtype": dtype}
        cell_params = {
            "kernel_size": kernel_size,
            "stride": 1,  # dilation only supported with stride=1
            "padding": "same",  # easier to handle growing dilation
            "groups": groups,
            "bias": bias,
            **factory_params,
        }
        cell_class = basecells.ResConvCell if residual_connections else basecells.ConvCell
        layer_sizes = [in_channels, *hidden_channels, out_channels]
        activations = [activation for _ in hidden_channels] + [out_activation]
        batch_norms = [batch_norm for _ in hidden_channels] + [out_batch_norm]
        for i, (in_size, out_size, act, bn) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:], activations, batch_norms)
        ):
            cell = cell_class(
                in_channels=in_size,
                out_channels=out_size,
                activation=act,
                batch_norm=bn,
                dilation=2 ** (i // 2),
                **cell_params,
            )
            self.append(cell)


class ConvEncoder(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int | None,
        out_features: int,
        kernel_size: int = 3,
        hidden_channels: Sequence[int] = (64, 64),
        hidden_features: Sequence[int] = (8, 8),
        activation_conv: torch.nn.Module = DEFAULT_ACTIVATION,
        activation_linear: torch.nn.Module = DEFAULT_ACTIVATION,
        activation_out: torch.nn.Module | None = None,
        residual_connections: bool = False,
        groups: int = 1,
        batch_norm: bool = True,
        out_batch_norm: bool = False,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_params = {"device": device, "dtype": dtype}
        self.append(
            ConvNet(
                in_channels=in_channels,
                out_channels=hidden_channels[-1],
                kernel_size=kernel_size,
                hidden_channels=hidden_channels[:-1],
                activation=activation_conv,
                out_activation=activation_conv,
                batch_norm=batch_norm,
                out_batch_norm=batch_norm,
                residual_connections=residual_connections,
                groups=groups,
                bias=bias,
                **factory_params,
            )
        )
        self.append(torch.nn.Flatten(start_dim=1))
        self.append(
            LinearNet(
                in_features=None,
                out_features=out_features,
                hidden_features=hidden_features,
                activation=activation_linear,
                out_activation=activation_out,
                batch_norm=batch_norm,
                out_batch_norm=out_batch_norm,
                bias=bias,
                **factory_params,
            )
        )
