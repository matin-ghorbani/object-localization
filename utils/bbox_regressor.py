import torch
from torch import nn


class ObjectDetector(nn.Module):
    def __init__(self, base_model, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.base_model = base_model
        self.num_classes = num_classes

        # Predict starting x-axis, starting y-axis, ending x-axis, ending y-axis
        self.regressor = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 4),
            nn.Sigmoid()
        )

        # Classifier for the object label
        self.classifier = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(512, self.num_classes),
        )

        self.base_model.fc = nn.Identity()

    def forward(self, x):
        features = self.base_model(x)

        bboxes = self.regressor(features)
        label = self.classifier(features)

        return bboxes, label
