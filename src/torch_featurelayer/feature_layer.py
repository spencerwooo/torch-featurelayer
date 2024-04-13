from typing import Any

import torch

from torch_featurelayer.rgetattr import rgetattr


class FeatureLayer:
    """Wraps a model and provides a hook to access the output of a feature layer.

    Feature layer paths are defined via dot notation:

    Args:
        model: The model containing the feature layer.
        feature_layer_path: The path to the feature layer in the model.

    Attributes:
        _model (torch.nn.Module): The model containing the feature layer.
        feature_layer_path (str): The path to the feature layer in the model.
        feature_layer_output (torch.Tensor): The output of the feature layer.
    """

    def __init__(self, model: torch.nn.Module, feature_layer_path: str):
        self._model: torch.nn.Module = model
        self.feature_layer_path: str = feature_layer_path
        self.feature_layer_output: torch.Tensor | None = None  # output of the feature layer (must be global)

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Perform a forward pass through the model and update the hooked feature layer.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: A tuple containing the feature layer output and the model output.
        """

        h: torch.utils.hooks.RemovableHandle | None = None  # hook handle

        def hook(module: torch.nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.feature_layer_output = output

        try:
            # Register hook
            layer = rgetattr(self._model, self.feature_layer_path)
            h = layer.register_forward_hook(hook)
        except AttributeError as e:
            raise AttributeError(f'Layer {self.feature_layer_path} not found in model.') from e

        # Forward pass and update hooked feature layer
        output: torch.Tensor = self._model(*args, **kwargs)

        # Remove hook
        if h is not None:
            h.remove()

        return self.feature_layer_output, output
