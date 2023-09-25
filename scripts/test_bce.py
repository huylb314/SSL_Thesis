import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Create a simple binary segmentation model
class BinarySegmentationModel(nn.Module):
    def __init__(self):
        super(BinarySegmentationModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return torch.sigmoid(x)

# Instantiate the model
model = BinarySegmentationModel()

# Create a synthetic example
input_image = torch.randn(1, 3, 200, 200)  # Batch size of 1, 3 channels, 200x200 image
ground_truth_mask = torch.randint(0, 2, (1, 1, 200, 200), dtype=torch.float32)  # Binary mask

# Forward pass
predicted_mask = model(input_image)

# Calculate the loss using F.binary_cross_entropy
loss = F.binary_cross_entropy(predicted_mask, ground_truth_mask)

# Backpropagation and optimization (not shown in this example)
