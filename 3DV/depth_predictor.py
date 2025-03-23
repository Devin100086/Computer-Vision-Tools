import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import io
import torch
from torchvision import transforms
from PIL import Image
from transformers import pipeline
import numpy as np

# 加载模型
model = pipeline(task="depth-estimation", model="depth-anything/depth-anything-V2-Base-hf")

def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored

def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

def transform(depth_pred):
        min_d = torch.min(depth_pred)
        max_d = torch.max(depth_pred)
        depth_pred = (depth_pred - min_d) / (max_d - min_d)

         # Convert to numpy
        depth_pred= depth_pred.cpu().numpy().astype(np.float32)

        # Resize back to original resolution
        pred_img = Image.fromarray(depth_pred)
        # pred_img = pred_img.resize(input_size)
        depth_pred = np.asarray(pred_img)

        # Clip output range: current size is the original size
        depth_pred = depth_pred.clip(0, 1)
        
        # colorization using the KITTI Color Plan.
        depth_pred_vis = depth_pred * 70
        disp_vis = 400/(depth_pred_vis+1e-3)
        disp_vis = disp_vis.clip(0,500)
    
        # Colorize
        depth_colored = colorize_depth_maps(
            depth_pred, 0, 1, cmap="Spectral"
        ).squeeze()  # [3, H, W], value in (0, 1)
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        depth_colored_img = Image.fromarray(depth_colored_hwc)
        return depth_colored_img

def predict_depth(image: Image.Image) -> np.ndarray:
    with torch.no_grad():
        depth_map = model(image)["depth"]
    depth_array = np.array(depth_map)
    output_image = Image.fromarray(depth_array.astype(np.uint8))

    depth_colored_img = transform(torch.from_numpy(depth_array))

    return output_image, depth_colored_img 

def depth_estimation(image: Image.Image) -> Image.Image:
    depth_map = predict_depth(image)
    return depth_map


with gr.Blocks() as demo:
    gr.Markdown("## Depth Anything V2")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type='pil', label="input image", height=400)
            with gr.Row():
                run_button = gr.Button("run")
        with gr.Column():
            output_rgb_image = gr.Image(type='pil', label="result", height=600)
            output_image = gr.Image(type='pil', label="result", height=600)

    run_button.click(fn=depth_estimation, inputs=input_image, outputs=[output_rgb_image, output_image])

demo.launch()