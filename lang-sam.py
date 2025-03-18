from lang_sam import LangSAM
import os
from PIL import Image
import cv2
import torch
import torchvision.utils as vutils
from torchvision.transforms.functional import to_tensor, gaussian_blur

mpath = "???"
model = LangSAM()
mnames = sorted(os.listdir(mpath))
for mp in mnames:
    image = Image.open(os.path.join(mpath, mp))
    mask = model.predict([image], ["truck"])[0]['masks']
    mask = torch.from_numpy(mask)
    # mask = gaussian_blur(mask, kernel_size=(77, 77))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    os.makedirs("???", exist_ok=True)
    vutils.save_image(mask.unsqueeze(0), os.path.join("???",mp.replace(".JPG", ".png")), nrow=1)
