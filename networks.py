from typing import Sequence
import torch
from .basecells import LazyConvCell, LazyLinearCell, ConvCell

class LazyLinearNet(torch.nn.Sequential):
    def __init__(
        self,
        out_features: int,
        hidden_features: Sequence[int] = (16,),
        activation: torch.nn.Module = torch.nn.ReLU(),
        **kwargs,  # arguments fowarded to LinearCell and torch.nn.Linear calls
    ):
        #create all but last cell
        cells = []
        for out_size in hidden_features:
            cells.append(LazyLinearCell(out_size, activation, **kwargs))
        # add the last cell, which has no activation
        cells.append(LazyLinearCell(out_features, None, **kwargs))
        super().__init__(*cells)
        

class LazyConvNet(torch.nn.Sequential):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: Sequence[int] = (16,),
        **kwargs,  # arguments fowarded to ConvCell and torch.nn.Conv2d calls
    ):
        # create all but last cell
        cells = []
        for out_size in hidden_channels:
            cells.append(LazyConvCell(out_size, **kwargs))
        # add last cell, which has no pooling
        cells.append(ConvCell(out_channels, pooling_params=None, **kwargs))
        super().__init__(*cells)


class LazyConvEncoder(torch.nn.Sequential):
    def __init__(
        self,
        hidden_channels: Sequence[int] = (16, 16),
        hidden_features: Sequence[int] = (16,),
        out_features: int = 16,
        activation_linear: torch.nn.Module = torch.nn.ReLU(),
        **kwargs,  # arguments fowarded to ConvCell and torch.nn.Conv2d calls
    ):
        modules = []
        try:
            # create the convolutional layers
            conv_layers = LazyConvNet(
                out_channels=hidden_channels[-1],
                hidden_channels=hidden_channels[:-1],
                **kwargs,
            )
            modules.append(conv_layers)
            
        except LazyConvCell.SqueezedDimError as e:
            msg, squeezed_dim = e.args
            if squeezed_dim > 2:
                raise e
            
        modules.append(torch.nn.Flatten(start_dim=1))

        linear_layers = LazyLinearNet(
            out_features=out_features,
            hidden_features=hidden_features,
            activation=activation_linear,
        )
        modules.append(linear_layers)

        super().__init__(*modules)
