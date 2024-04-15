import torch
from torch_featurelayer import FeatureLayers
from torchvision.models import resnet50

# Load a pretrained ResNet-50 model
model = resnet50(weights='DEFAULT').eval()

# Hook onto layers ['layer1', 'layer2', 'layer3', 'layer4'] of the model
layer_paths = ['layer1', 'layer2', 'layer3', 'layer4']
hooked_model = FeatureLayers(model, layer_paths)

# Forward pass an input tensor through the model
x = torch.randn(1, 3, 224, 224)
feature_outputs, output = hooked_model(x)

# Print the output shapes
for layer_path, feature_output in feature_outputs.items():
    print(f'{layer_path} output shape: {feature_output.shape}')
