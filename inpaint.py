import gradio as gr
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
import numpy as np

model = "stabilityai/stable-diffusion-2-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # 如果有GPU，使用cuda

def inpaint(image, mask, prompt):
    H,W = image['image'].size
    image['image'] = image['image'].resize((512,512))
    mask = mask.resize((512,512))
    generator = torch.Generator(device="cuda").manual_seed(1)
    result = pipe(prompt=prompt, image=image['image'], mask_image=mask,
                  num_inference_steps=20, generator=generator).images[0]
    result = result.resize((H, W))
    return result

def save_button(image, mask):

    mask = mask['mask'].convert("L")

    mask_array = np.array(mask)
    binary_mask = np.where(mask_array > 128, 255, 0).astype(np.uint8)
    binary_mask_image = Image.fromarray(binary_mask)

    binary_mask_image.save("mask.png")
    return binary_mask_image

with gr.Blocks() as demo:
    gr.Markdown("## Stable Diffusion 2 Inpainting Demo")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type='pil', tool="sketch", label="Input Image",height=400)
            generate_button = gr.Button("generate mask")
            with gr.Row():
                mask_image = gr.Image(type='pil',label="Mask Image", interactive=True, show_download_button=True, height=500)
        with gr.Column():
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Row():
                output_image = gr.Image(type='pil',label="Output Image")

    generate_button.click(fn=save_button, inputs=[input_image, input_image], outputs=mask_image)
    run_button.click(fn=inpaint, inputs=[input_image, mask_image, prompt], outputs=output_image)

demo.launch()
