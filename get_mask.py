import gradio as gr
import numpy as np
from PIL import Image

def save_mask(image, mask):

    mask = mask['mask'].convert("L")

    mask_array = np.array(mask)
    binary_mask = np.where(mask_array > 128, 255, 0).astype(np.uint8)
    binary_mask_image = Image.fromarray(binary_mask)

    binary_mask_image.save("mask.png")
    return binary_mask_image

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(source="upload", tool="sketch", type="pil", label="上传图片并涂鸦", height=400)
            save_button = gr.Button("save mask")
    with gr.Row():
        with gr.Column(scale=1):
            pass 
        with gr.Column(scale=2, min_width=400):
            mask_output = gr.Image(label="generate mask", height=800, width=800)
        with gr.Column(scale=1):
            pass

    save_button.click(fn=save_mask, inputs=[image_input, image_input], outputs=mask_output)

demo.launch()
