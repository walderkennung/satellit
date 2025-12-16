import logging

import satellit_sam.pytorch as pytorch
from segment_anything import sam_model_registry

logging.basicConfig(level=logging.DEBUG)

pytorch_instance = pytorch.init()
pytorch_instance.debug_info()

sam = sam_model_registry["vit_h"](checkpoint="models/sam/sam_vit_h_4b8939.pth")
sam.to(pytorch_instance.device)
print("SAM loaded successfully on", pytorch_instance.device)
