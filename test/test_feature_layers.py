import pytest
import torch
from torch_featurelayer import FeatureLayers
from torchvision.models import alexnet


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


def test_feature_layers_invalid_model():
    with pytest.raises(AssertionError) as e:
        model = 'invalid_model'
        _ = FeatureLayers(model, ['features.2', 'features.5', 'features.9'])
    assert str(e.value) == 'model must be a torch.nn.Module'


def test_feature_layers_contain_nonexistent_layer_path():
    model = alexnet()
    hooked_model = FeatureLayers(model, ['features.1', 'path.to.blah'])

    x = torch.randn(1, 3, 224, 224)
    feature_outputs, outputs = hooked_model(x)

    assert feature_outputs['features.1'] is not None
    assert feature_outputs['path.to.blah'] is None  # nonexistent layer path ignored
    assert outputs is not None


def test_feature_layers_empty_layer_paths():
    model = alexnet()
    hooked_model = FeatureLayers(model, [])

    x = torch.randn(1, 3, 224, 224)
    feature_outputs, output = hooked_model(x)

    assert feature_outputs == {}
    assert output is not None
