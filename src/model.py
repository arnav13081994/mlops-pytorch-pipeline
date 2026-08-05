from torch import nn


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 1x28x28
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Out: 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Out: 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Out: 64x14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Out: 64x7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),  # 10 classes
        )

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return logits


def get_model() -> nn.Module:
    return NeuralNetwork()
