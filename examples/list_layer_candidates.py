from torchvision.models import resnet18
from torch_featurelayer import get_layer_candidates

# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     ...
#   )
#   (layer2): Sequential( ... )
#   (layer3): Sequential( ... )
#   (layer4): Sequential( ... )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Linear(in_features=512, out_features=1000, bias=True)
# )
model = resnet18(weights='DEFAULT').eval()
layer_paths = get_layer_candidates(model, max_depth=2)  # this will return a generator

# ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer1.0', 'layer1.1', 'layer2', 'layer2.0', 'layer2.1', 'layer3',
# 'layer3.0', 'layer3.1', 'layer4', 'layer4.0', 'layer4.1', 'avgpool', 'fc']
print(list(layer_paths))
