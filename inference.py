import torch
import torch.nn as nn
import timm

DEVICE = torch.device("cpu")

NUM_CLASSES = 4  

class SwinClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=NUM_CLASSES
        )

    def forward(self, x):
        return self.model(x)


def load_model(model_path):
    model = SwinClassifier()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model
