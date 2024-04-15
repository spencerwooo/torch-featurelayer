import torch
from torchvision.models import resnet50

from torch_featurelayer import FeatureLayers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a pretrained ResNet-50 model
model = resnet50(weights='DEFAULT').eval().to(device)

# Hook onto layers ['layer1', 'layer2', 'layer3', 'layer4'] of the model
layer_paths = ['layer1', 'layer2', 'layer3', 'layer4']
hooked_model = FeatureLayers(model, layer_paths)

# Forward pass an input tensor through the model
x = torch.randn(1, 3, 224, 224).to(device)
feature_outputs, output = hooked_model(x)

# Print the output shapes
for layer_path, feature_output in feature_outputs.items():
    print(f'{layer_path} output shape: {feature_output.shape}')
