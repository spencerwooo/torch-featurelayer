import torch
from torchvision.models import alexnet
from torch_featurelayer import FeatureLayer


def test_feature_layer():
    model = alexnet()
    hooked_model = FeatureLayer(model, 'features.12')

    x = torch.randn(1, 3, 224, 224)
    feature_output, output = hooked_model(x)

    assert feature_output.shape == (1, 256, 6, 6)
    assert output.shape == (1, 1000)
