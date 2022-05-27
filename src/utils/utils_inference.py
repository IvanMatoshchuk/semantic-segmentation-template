import torch
from monai.inferers import SlidingWindowInferer
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import numpy as np

from src.model.segmentation_model import HoneyBeeModel

infer = SlidingWindowInferer(roi_size=(1024, 1024), overlap=0.2, mode="gaussian")


def load_model(checkpoint_path: str, device: str = "cpu") -> HoneyBeeModel:
    pretrained_model = HoneyBeeModel.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=device)
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()
    pretrained_model.freeze()
    print("model loaded!")
    return pretrained_model


def make_single_prediction(image: np.ndarray, model: HoneyBeeModel, device: str = "cpu") -> torch.tensor:

    transforms = A.Compose([A.Normalize(mean=0, std=1), ToTensorV2()])

    image = transforms(image=image)["image"]
    # print("\nImage after transform: ", image.size(), "\n")
    image = image.unsqueeze(0)
    # pred_logits = model(image.to(device))
    pred_logits = infer(image.to(device), model)

    # pred = torch.argmax(pred_logits.squeeze(), dim=0).detach().cpu().numpy()

    return pred_logits
