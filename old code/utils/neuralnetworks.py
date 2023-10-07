from __future__ import annotations
import torch
import math
from dataclasses import dataclass
from typing import Any, TypeVar, Protocol, Iterable, Optional

# region protocols


class NeuralNetwork(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, batched=True)

    def forward(self, x: torch.Tensor, batched: bool) -> torch.Tensor:
        ...


NNType = TypeVar("NNType", bound=NeuralNetwork, covariant=True)


class NNFactory(Protocol[NNType]):
    def __call__(
        self, input_shape: tuple[int, ...], output_shape: tuple[int, ...]
    ) -> NNType:
        return self.make(input_shape, output_shape)

    def make(
        self, input_shape: tuple[int, ...], output_shape: tuple[int, ...]
    ) -> NNType:
        ...



class OptimizableModule(torch.nn.Module):
    def set_optimizer(self, lr: float = 0.001) -> None:
        self._optimizer = torch.optim.Adam(params=self.parameters(recurse=True), lr=lr)

    def optimization_step(self, loss: torch.Tensor) -> None:
        # zero gradients
        self._optimizer.zero_grad()
        # compute gradients
        loss.backward()  # type: ignore
        # update weights
        self._optimizer.step()


class OptimizableNN(NeuralNetwork, Protocol):
    def optimization_step(self, loss: torch.Tensor) -> None:
        ...


# endregion


# region linear nets


class OptimizableLinearNet(OptimizableModule):
    def __init__(
        self,
        Nin: int,
        Nout: int,
        Nh: Iterable[int] = (64,),
        activation: torch.nn.Module = torch.nn.ReLU(),
        output_activation: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()

        # create layers starting with flatten
        layers: list[torch.nn.Module] = [torch.nn.Flatten(start_dim=1)]
        # alternate linear layers and activation functions
        for size_in, size_out in zip([Nin] + list(Nh), list(Nh) + [Nout]):
            layers.append(torch.nn.Linear(size_in, size_out))
            layers.append(activation)
        # remove last activation
        layers.pop()

        # add output activation
        if output_activation is not None:
            layers.append(output_activation)

        # create sequential model
        self._layers = torch.nn.Sequential(*layers)

        self.set_optimizer()

    def forward(self, x: torch.Tensor, batched: Optional[bool] = True) -> torch.Tensor:
        if not batched:
            x = x.unsqueeze(0)
        return self._layers(x)


@dataclass(frozen=True, eq=False)
class OptimizableLinearNetFactory(NNFactory[OptimizableLinearNet]):
    Nh: Iterable[int] = (64,)
    activation = torch.nn.ReLU()
    output_activation: Optional[torch.nn.Module] = None

    def make(
        self, input_shape: tuple[int, ...], output_shape: tuple[int, ...]
    ) -> OptimizableLinearNet:
        return OptimizableLinearNet(
            Nin=math.prod(input_shape),
            Nout=math.prod(output_shape),
            Nh=self.Nh,
            activation=self.activation,
            output_activation=self.output_activation,
        )


# endregion
