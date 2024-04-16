import pytest
import torch
from torch_featurelayer import FeatureLayer
from torchvision.models import alexnet


def test_feature_layer():
    model = alexnet()
    hooked_model = FeatureLayer(model, 'features.12')

    x = torch.randn(1, 3, 224, 224)
    feature_output, output = hooked_model(x)

    assert feature_output.shape == (1, 256, 6, 6)
    assert output.shape == (1, 1000)


def test_feature_layer_invalid_model():
    with pytest.raises(AssertionError) as e:
        model = 'invalid_model'
        _ = FeatureLayer(model, 'features.12')

    assert str(e.value) == 'model must be a torch.nn.Module'


def test_feature_layer_nonexistent_layer_path():
    with pytest.raises(AttributeError) as e:
        model = alexnet()
        hooked_model = FeatureLayer(model, 'path.to.blah')

        x = torch.randn(1, 3, 224, 224)
        _ = hooked_model(x)

    assert str(e.value) == 'Layer path.to.blah not found in model.'
