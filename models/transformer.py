from models.vit import ViTForImageClassification
from transformers import ViTFeatureExtractor
import torch


class Cifar10Transformer(torch.nn.Module):
    def __init__(self, device):
        super(Cifar10Transformer, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
        self.classifier = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10').eval()
        self.classifier.to(device)
        self.device = device

    def forward(self, x):
        x = self.feature_extractor(torch.unbind(x.to('cpu')), return_tensors="pt")["pixel_values"]
        x = self.classifier(x.to(self.device))
        return x.logits

# transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
