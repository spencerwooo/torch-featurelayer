[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![lint](https://github.com/spencerwooo/torch-featurelayer/actions/workflows/ci.yml/badge.svg)](https://github.com/spencerwooo/torch-featurelayer/actions/workflows/ci.yml)

# torch-featurelayer

🧠 Simple utility functions and wrappers for hooking onto layers within PyTorch models for feature extraction.

> [!TIP]
> This library is intended to be a simplified and well-documented implementation for extracting a PyTorch model's intermediate layer output(s). For a more sophisticated and complete implementation, either consider using [`torchvision.models.feature_extraction`](https://pytorch.org/vision/stable/feature_extraction.html), or check the official [`torch.fx`](https://pytorch.org/docs/stable/fx.html). 

## Usage

```python
import torch
from torchvision.models import vgg11

from torch_featurelayer import FeatureLayer

# Load a pretrained VGG-11 model
model = vgg11(weights='DEFAULT').eval()

# Hook onto layer `features.15` of the model
layer_path = 'features.15'
hooked_model = FeatureLayer(model, layer_path)

# Forward pass an input tensor through the model
x = torch.randn(1, 3, 224, 224)
feature_output, output = hooked_model(x)

# Print the output shape
print(f'Feature layer output shape: {feature_output.shape}')  # [1, 512, 14, 14]
print(f'Model output shape: {output.shape}')  # [1, 1000]
```

Check the [examples](./examples/) directory for more.

## API

> `torch_featurelayer.FeatureLayer(model: torch.nn.Module, feature_layer_path: str)`

> `torch_featurelayer.FeatureLayers(model: torch.nn.Module, feature_layer_paths: list[str])`

> `torch_featurelayer.get_layer_candidates(module: nn.Module, max_depth: int = 1)`

## License

[MIT](./LICENSE)