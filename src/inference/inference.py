# %%
# TODO add argparse
import os
import sys
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

project_path = Path(__file__).parent.parent
print(project_path)
sys.path.append(project_path)
sys.path.append(os.path.join(project_path, "src"))


from model.model import UnetModel
from datamodule.custom_datasets import CustomDataset
from utils.utils import read_config

project_path = Path(__file__).parent.parent
path_to_config = os.path.join(project_path, "config", "datamodule.yaml")
data_config = read_config(path_to_config)
# %%
version = "version_49"
checkpoint_path = os.path.join(project_path, "lightning_logs", f"{version}", "checkpoints", "epoch=1-step=319.ckpt")


pretrained_model = UnetModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
pretrained_model = pretrained_model.to("cuda")
pretrained_model.eval()
pretrained_model.freeze()


bees_data = CustomDataset(**data_config["dataset"]["val"], phase="val")

# x = iter(bees_data.val_dataloader())
# i, j = next(x)
image, mask = bees_data[0]
print("\n *** Image size: ", image.unsqueeze(0).size(), "\n")
# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
fig.suptitle("predicted_mask//original_mask")

batch_preds = torch.sigmoid(pretrained_model(image.unsqueeze(0).to("cuda")))
batch_preds = torch.argmax(batch_preds.squeeze(), dim=0)

print(batch_preds.size())

batch_preds = batch_preds.detach().cpu().numpy()  # convert tensort into numpy
ax1.imshow(
    np.squeeze(batch_preds > 0.5), cmap="gray"
)  # Remove single-dimensional entries from the array. Setting threshold to 0.5
ax2.imshow(np.squeeze(mask), cmap="gray")
#%%

path_to_save = os.path.join(project_path, "inference", "output", "comparison.png")
plt.show()
plt.savefig(path_to_save)
