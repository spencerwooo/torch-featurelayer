[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![image](https://img.shields.io/pypi/v/torch-featurelayer.svg)](https://pypi.python.org/pypi/torch-featurelayer)
[![image](https://img.shields.io/pypi/l/torch-featurelayer.svg)](https://pypi.python.org/pypi/torch-featurelayer)
[![image](https://img.shields.io/pypi/pyversions/torch-featurelayer.svg)](https://pypi.python.org/pypi/torch-featurelayer)
[![lint](https://github.com/spencerwooo/torch-featurelayer/actions/workflows/ci.yml/badge.svg)](https://github.com/spencerwooo/torch-featurelayer/actions/workflows/ci.yml)

# torch-featurelayer

🧠 Simple utility functions and wrappers for hooking onto layers within PyTorch models for feature extraction.

> [!TIP]
> This library is intended to be a simplified and well-documented implementation for extracting a PyTorch model's intermediate layer output(s). For a more sophisticated and complete implementation, either consider using [`torchvision.models.feature_extraction`](https://pytorch.org/vision/stable/feature_extraction.html), or check the official [`torch.fx`](https://pytorch.org/docs/stable/fx.html).

## Install

```shell
pip install torch-featurelayer
```

## Usage

Imports:

```python
import torch
from torchvision.models import vgg11
from torch_featurelayer import FeatureLayer
```

Load a pretrained VGG-11 model:

```python
model = vgg11(weights='DEFAULT').eval()
```

Hook onto layer `features.15` of the model:

```python
layer_path = 'features.15'
hooked_model = FeatureLayer(model, layer_path)
```

Forward pass an input tensor through the model:

```python
x = torch.randn(1, 3, 224, 224)
feature_output, output = hooked_model(x)
```

`feature_output` is the output of layer `features.15`. Print the output shape:

```python
print(f'Feature layer output shape: {feature_output.shape}')  # [1, 512, 14, 14]
print(f'Model output shape: {output.shape}')  # [1, 1000]
```

Check the [examples](./examples/) directory for more.

## API

### `torch_featurelayer.FeatureLayer`

The `FeatureLayer` class wraps a model and provides a hook to access the output of a specific feature layer.

- `__init__(self, model: torch.nn.Module, feature_layer_path: str)`

    Initializes the `FeatureLayer` instance.

    - `model`: The model containing the feature layer.
    - `feature_layer_path`: The path to the feature layer in the model.

- `__call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor | None, torch.Tensor]`

    Performs a forward pass through the model and updates the hooked feature layer.

    - `*args`: Variable length argument list.
    - `**kwargs`: Arbitrary keyword arguments.

    Returns a tuple containing the feature layer output and the model output.

### `torch_featurelayer.FeatureLayers`

The `FeatureLayers` class wraps a model and provides hooks to access the output of multiple feature layers.

- `__init__(self, model: torch.nn.Module, feature_layer_paths: list[str])`

    Initializes the `FeatureLayers` instance.

    - `model`: The model containing the feature layers.
    - `feature_layer_paths`: A list of paths to the feature layers in the model.

- `__call__(self, *args: Any, **kwargs: Any) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]`

    Performs a forward pass through the model and updates the hooked feature layers.

    - `*args`: Variable length argument list.
    - `**kwargs`: Arbitrary keyword arguments.

    Returns a tuple containing the feature layer outputs and the model output.

### `torch_featurelayer.get_layer_candidates(module: torch.nn.Module, max_depth: int = 1) -> Generator[str, None, None]`

The `get_layer_candidates` function returns a generator of layer paths for a given model up to a specified depth.

- `model`: The model to get layer paths from.
- `max_depth`: The maximum depth to traverse in the model's layers.

Returns a generator of layer paths.

## License

[MIT](./LICENSE)
