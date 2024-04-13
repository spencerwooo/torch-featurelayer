import torch
from torchvision.models import vgg11

from torch_featurelayer import FeatureLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a pretrained VGG-11 model
model = vgg11(weights='DEFAULT').eval().to(device)

# Hook onto layer `features.15` of the model
layer_path = 'features.15'
hooked_model = FeatureLayer(model, layer_path)

# Forward pass an input tensor through the model
x = torch.randn(1, 3, 224, 224).to(device)
feature_output, output = hooked_model(x)

# Print the output shape
print(f'Feature layer output shape: {feature_output.shape}')  # [1, 512, 14, 14]
print(f'Model output shape: {output.shape}')  # [1, 1000]
