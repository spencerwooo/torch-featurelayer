from typing import Any

import torch

from torch_featurelayer.rgetattr import rgetattr


class FeatureLayers:
    """Wraps a model and provides hooks to access the output of multiple feature layers.

    Args:
        model: The model containing the feature layer.
        feature_layer_paths: A list of paths to the feature layers in the model.

    Attributes:
        _model (torch.nn.Module): The model containing the feature layer.
        feature_layer_paths (list[str]): A list of paths to the feature layers in the model.
        feature_layer_outputs (list[torch.Tensor]): The output of the feature layers.
    """

    def __init__(self, model: torch.nn.Module, feature_layer_paths: list[str]):
        self._model: torch.nn.Module = model
        self.feature_layer_paths: list[str] = feature_layer_paths
        self.feature_layer_outputs: dict[str, torch.Tensor | None] = dict().fromkeys(feature_layer_paths, None)

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        """Perform a forward pass through the model and update the hooked feature layers.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: A tuple containing the feature layer outputs and the model output.
        """

        hs = []  # hook handles

        for feature_layer_path in self.feature_layer_paths:

            def hook(
                module: torch.nn.Module,
                input: torch.Tensor,
                output: torch.Tensor,
                feature_layer_path=feature_layer_path,
            ):
                self.feature_layer_outputs[feature_layer_path] = output

            try:
                # Register hook
                layer = rgetattr(self._model, feature_layer_path)
                h = layer.register_forward_hook(hook)
                hs.append(h)
            except AttributeError:
                # skip hook register if layer not found
                print(f'Warning: Layer {feature_layer_path} not found in model, skipping hook register.')
                continue

        # Forward pass and update hooked feature layers
        output: torch.Tensor = self._model(*args, **kwargs)

        # Remove hooks
        for h in hs:
            h.remove()

        return self.feature_layer_outputs, output
