import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# (Optional) If you truly need skimage-based file loading for standalone use, keep these:
# from skimage.io import imread
# from skimage import img_as_ubyte, transform

# ===============================
# (Optional) Image preprocessing helper (unused in Streamlit flow)
# ===============================
def load_signature(path, size=(155, 220)):
    """Load image as grayscale, resize, convert to tensor, normalize."""
    # This path uses PIL-only flow to avoid skimage dependency at runtime
    img = Image.open(path).convert("L")
    img = img.resize((size[1], size[0]))  # PIL expects (W,H)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return preprocess(img).unsqueeze(0)  # [1,1,H,W]

# ===============================
# Utility layers
# ===============================
def conv_bn_mish(in_channels, out_channels, kernel_size, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        ("bn", nn.BatchNorm2d(out_channels)),
        ("act", nn.Mish())
    ]))

def linear_bn_mish(in_features, out_features):
    return nn.Sequential(OrderedDict([
        ("fc", nn.Linear(in_features, out_features, bias=False)),
        ("bn", nn.BatchNorm1d(out_features)),
        ("act", nn.Mish())
    ]))

# ===============================
# SigNet Architecture
# ===============================
class SigNet(nn.Module):
    """SigNet model for signature verification."""
    def __init__(self):
        super().__init__()
        self.feature_space_size = 2048

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1',   conv_bn_mish(1,   96, 11, stride=4)),
            ('maxpool1', nn.MaxPool2d(3, 2)),
            ('conv2',   conv_bn_mish(96,  256, 5, pad=2)),
            ('maxpool2', nn.MaxPool2d(3, 2)),
            ('conv3',   conv_bn_mish(256, 384, 3, pad=1)),
            ('conv4',   conv_bn_mish(384, 384, 3, pad=1)),
            ('conv5',   conv_bn_mish(384, 256, 3, pad=1)),
            ('maxpool3', nn.MaxPool2d(3, 2)),
        ]))

        # For input 1x155x220, the conv stack yields 256 x 3 x 5 (as you had)
        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', linear_bn_mish(256 * 3 * 5, 2048)),
            ('fc2', linear_bn_mish(2048, 2048)),
        ]))

    def forward_once(self, img):
        x = self.conv_layers(img)
        x = x.view(x.shape[0], 256 * 3 * 5)
        x = self.fc_layers(x)
        return x

    def forward(self, img1, img2):
        return self.forward_once(img1), self.forward_once(img2)

# ===============================
# Model loader
# ===============================
def _strip_dataparallel_prefix(state_dict):
    """Remove 'module.' prefix if present (from DataParallel)."""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def load_model(weights_path, device):
    """
    Robust loader:
    - Works on CPU/GPU via map_location
    - Accepts checkpoints like:
        * raw state_dict
        * {'model': state_dict, ...}
        * DataParallel-wrapped keys ('module.*')
    """
    model = SigNet().to(device)
    checkpoint = torch.load(weights_path, map_location=device)

    # Extract state dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint  # assume raw state_dict

    state_dict = _strip_dataparallel_prefix(state_dict)

    # Load with strict=True to catch real mismatches; relax to False if needed
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        # Fallback to strict=False if minor buffers mismatch
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model
