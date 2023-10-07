from typing import Sequence
import torch
from . import basecells


class LinearNet(torch.nn.Sequential):
    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        hidden_features: Sequence[int] = (16,),
        activation: torch.nn.Module = torch.nn.ReLU(),
        out_activation: torch.nn.Module | None = None,
        batch_norm: bool = True,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        cell_params = {
            "batch_norm": batch_norm,
            "bias": bias,
            "device": device,
            "dtype": dtype,
        }

        layer_sizes = [in_features, *hidden_features, out_features]
        for in_size, out_size in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            self.append(
                basecells.LinearCell(
                    in_features=in_size,
                    out_features=out_size,
                    activation=activation,
                    **cell_params,
                )
            )

        # last cell has no activation
        self.append(
            basecells.LinearCell(
                in_features=layer_sizes[-2],
                out_features=layer_sizes[-1],
                activation=out_activation,
                **cell_params,
            )
        )


class ConvNet(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int | None,
        out_channels: int,
        kernel_size: int = 3,
        hidden_channels: Sequence[int] = (16,),
        activation: torch.nn.Module = torch.nn.ReLU(),
        batch_norm: bool = True,
        residual_connections: bool = False,
        groups: int = 1,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        cell_params = {
            "kernel_size": kernel_size,
            "activation": activation,
            "batch_norm": batch_norm,
            "stride": 1,  # dilation only supported with stride=1
            "padding": "same",  # easier to handle growing dilation
            "groups": groups,
            "bias": bias,
            "device": device,
            "dtype": dtype,
        }

        layer_sizes = [in_channels, *hidden_channels, out_channels]
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            if residual_connections:
                self.append(
                    basecells.ResConvCell(
                        in_channels=in_size,
                        out_channels=out_size,
                        dilation=2**i,
                        **cell_params,
                    )
                )
            else:
                self.append(
                    basecells.ConvCell(
                        in_channels=in_size,
                        out_channels=out_size,
                        dilation=2**i,
                        **cell_params,
                    )
                )


class ConvEncoder(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int | None,
        out_features: int,
        kernel_size: int = 3,
        hidden_channels: Sequence[int] = (16, 16),
        hidden_features: Sequence[int] = (16,),
        activation_conv: torch.nn.Module = torch.nn.ReLU(),
        activation_linear: torch.nn.Module = torch.nn.ReLU(),
        residual_connections: bool = False,
        groups: int = 1,
        batch_norm: bool = True,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self._initialized = False

        self.append(basecells.Squeeze(from_dim=2))

        self.append(
            ConvNet(
                in_channels=in_channels,
                out_channels=hidden_channels[-1],
                kernel_size=kernel_size,
                hidden_channels=hidden_channels[:-1],
                activation=activation_conv,
                batch_norm=batch_norm,
                residual_connections=residual_connections,
                groups=groups,
                bias=bias,
                device=device,
                dtype=dtype,
            )
        )

        self.append(torch.nn.Flatten(start_dim=1))

        self.append(
            LinearNet(
                in_features=None,
                out_features=out_features,
                hidden_features=hidden_features,
                activation=activation_linear,
                batch_norm=batch_norm,
                bias=bias,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            if self[0].forward(input).ndim <= 2:
                self.pop(0)
                self.pop(0)
            self._initialized = True
        return super().forward(input)
