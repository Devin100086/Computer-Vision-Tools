import gradio as gr
import matplotlib.pyplot as plt
import io
import torch
from torchvision import transforms
from PIL import Image
from transformers import pipeline
import numpy as np

# 加载模型
model = pipeline(task="depth-estimation", model="depth-anything/depth-anything-V2-Base-hf")


def predict_depth(image: Image.Image) -> np.ndarray:
    # 预处理
    with torch.no_grad():
        depth_map = model(image)["depth"]
    depth_array = np.array(depth_map)

    colormap = plt.get_cmap('plasma')
    colored_array = colormap(depth_array / 255.0)[:, :, :3] 
    colored_array = (colored_array * 255).astype(np.uint8)
    rgb_depth_image = Image.fromarray(colored_array)
    return rgb_depth_image

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
            output_image = gr.Image(type='pil', label="result", height=700)

    run_button.click(fn=depth_estimation, inputs=input_image, outputs=output_image)

demo.launch()