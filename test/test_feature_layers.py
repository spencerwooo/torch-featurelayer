import torch
from torchvision.models import alexnet
from torch_featurelayer import FeatureLayers


def test_feature_layers():
    model = alexnet()
    layer_paths = [
        'features.2',
        'features.5',
        'features.9',
        'features.12',
        'avgpool',
        'classifier.2',
        'classifier.4',
    ]
    hooked_model = FeatureLayers(model, layer_paths)

    x = torch.randn(1, 3, 224, 224)
    feature_outputs, output = hooked_model(x)

    feature_output_shapes = {
        'features.2': (1, 64, 27, 27),
        'features.5': (1, 192, 13, 13),
        'features.9': (1, 256, 13, 13),
        'features.12': (1, 256, 6, 6),
        'avgpool': (1, 256, 6, 6),
        'classifier.2': (1, 4096),
        'classifier.4': (1, 4096),
    }
    for layer_path, feature_output in feature_outputs.items():
        assert feature_output.shape == feature_output_shapes[layer_path]
    assert output.shape == (1, 1000)
