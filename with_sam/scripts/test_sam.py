import torch
from segment_anything import sam_model_registry

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Using device: {device}")

sam = sam_model_registry["vit_h"](checkpoint="models/sam/sam_vit_h_4b8939.pth")
sam.to(device)
print("SAM loaded successfully on", device)
