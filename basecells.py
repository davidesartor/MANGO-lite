from typing import Any, Sequence
import torch
from torch import nn


class Squeeze(nn.Module):
    def __init__(self, dim: int | None = None):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.squeeze(x, dim=self.dim)
        if x.shape[0] == 1:
            y = torch.unsqueeze(y, dim=0)
        return y


class Unsqueeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.unsqueeze(x, dim=self.dim)


class LinearCell(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module | None = nn.ReLU(),
        batch_norm: bool = True,
        **kwargs,  # arguments fowarded to torch.nn.Linear call
    ):
        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm1d(in_features))
        layers.append(nn.Linear(in_features, out_features, **kwargs))
        if activation is not None:
            layers.append(activation)
        super().__init__(*layers)


class LazyLinearCell(nn.Sequential):
    def __init__(
        self,
        out_features: int,
        activation: nn.Module | None = nn.ReLU(),
        batch_norm: bool = True,
        **kwargs,  # arguments fowarded to torch.nn.Linear call
    ):
        layers = []
        if batch_norm:
            layers.append(nn.LazyBatchNorm1d())
        layers.append(torch.nn.LazyLinear(out_features, **kwargs))
        if activation is not None:
            layers.append(activation)
        super().__init__(*layers)


class ConvCell(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] | tuple[int, int, int] = (3, 3),
        activation: torch.nn.Module | None = torch.nn.ReLU(),
        pooling_params: dict[str, Any] | None = {"kernel_size": 2, "stride": 2},
        batch_norm: bool = True,
        **kwargs,  # arguments fowarded to torch.nn.ConvXd call
    ):
        layers = []

        if isinstance(kernel_size, int):
            bn = nn.BatchNorm1d(in_channels)
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
            pl = nn.MaxPool1d(**pooling_params) if pooling_params else None
        elif len(kernel_size) == 2:
            bn = nn.BatchNorm2d(in_channels)
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
            pl = nn.MaxPool2d(**pooling_params) if pooling_params else None
        elif len(kernel_size) == 3:
            bn = nn.BatchNorm3d(in_channels)
            conv = nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)
            pl = nn.MaxPool3d(**pooling_params) if pooling_params else None
        else:
            raise ValueError(f"Invalid kernel dimension: {kernel_size}")

        if batch_norm:
            layers.append(bn)
        layers.append(conv)
        if pooling_params is not None:
            layers.append(pl)
        if activation is not None:
            layers.append(activation)

        super().__init__(*layers)


class LazyConvCell(torch.nn.Sequential):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        activation: torch.nn.Module | None = torch.nn.ReLU(),
        pooling_params: dict[str, Any] | None = {"kernel_size": 2, "stride": 2},
        batch_norm: bool = True,
        **kwargs,  # arguments fowarded to torch.nn.ConvXd call
    ):
        self.initialization_params = {
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "activation": activation,
            "pooling_params": pooling_params,
            "batch_norm": batch_norm,
            "kwargs": kwargs,
        }
        
    class SqueezedDimError(ValueError):
            pass

    def lazy_init(
        self,
        first_input: torch.Tensor,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        activation: torch.nn.Module | None = torch.nn.ReLU(),
        pooling_params: dict[str, Any] | None = {"kernel_size": 2, "stride": 2},
        batch_norm: bool = True,
        **kwargs,
    ):
        layers = []

        squeeze_layer = Squeeze()
        layers.append(squeeze_layer)

        squeezed_dim = squeeze_layer(first_input).ndim
        if first_input.shape[1] == 1:
            squeezed_dim += 1
            layers.append(Unsqueeze(0))

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
            
        

        if squeezed_dim == 2:
            bn = nn.LazyBatchNorm1d()
            conv = nn.LazyConv1d(out_channels, kernel_size[0], **kwargs)
            pl = nn.MaxPool1d(**pooling_params) if pooling_params else None
        elif squeezed_dim == 3:
            bn = nn.LazyBatchNorm2d()
            conv = nn.LazyConv2d(out_channels, kernel_size[:2], **kwargs)
            pl = nn.MaxPool2d(**pooling_params) if pooling_params else None
        elif squeezed_dim == 4:
            bn = nn.LazyBatchNorm3d()
            conv = nn.LazyConv3d(out_channels, kernel_size, **kwargs)
            pl = nn.MaxPool3d(**pooling_params) if pooling_params else None
        else:
            raise self.SqueezedDimError(f"Invalid (squeezed) input dimension: ", squeezed_dim)

        if batch_norm:
            layers.append(bn)
        layers.append(conv)
        if pooling_params is not None:
            layers.append(pl)
        if activation is not None:
            layers.append(activation)

        super().__init__(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.initialization_params is not None:
            self.lazy_init(x, **self.initialization_params)
            self.initialization_params = None
        return super().forward(x)
