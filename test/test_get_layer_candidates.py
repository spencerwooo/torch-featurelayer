from torchvision.models import alexnet
from torch_featurelayer import get_layer_candidates


def test_get_layer_candidates_depth1():
    model = alexnet()
    candidates = list(get_layer_candidates(model, max_depth=1))
    assert candidates == [
        'features',
        'avgpool',
        'classifier',
    ]


def test_get_layer_candidates_depth2():
    model = alexnet()
    candidates = list(get_layer_candidates(model, max_depth=2))
    assert candidates == [
        'features',
        'features.0',
        'features.1',
        'features.2',
        'features.3',
        'features.4',
        'features.5',
        'features.6',
        'features.7',
        'features.8',
        'features.9',
        'features.10',
        'features.11',
        'features.12',
        'avgpool',
        'classifier',
        'classifier.0',
        'classifier.1',
        'classifier.2',
        'classifier.3',
        'classifier.4',
        'classifier.5',
        'classifier.6',
    ]
