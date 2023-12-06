import gradio as gr
import diffusers
import torch


def load_pipeline(model):
    pipe = diffusers.DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")



def diffusion(prompt_positive, prompt_negative):
    pipe = diffusers.DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("mps")
    image = pipe(prompt=prompt_positive, num_inference_steps=1, guidance_scale=0.0).images[0]
    return image



with gr.Blocks() as demo:
    gr.Markdown("<h1> Hallo Welt <h1>")
    model = gr.Dropdown(choices=["stabilityai/sdxl-turbo", ],label="model")
    out = model.change(load_pipeline, inputs=model,show_progress=True)

    prompt_positive = gr.Text(label="Positive Prompt")
    prompt_negative = gr.Text(label="Negative Prompt")

    result = gr.Image()

    btn = gr.Button("Run")
    btn.click(diffusion,inputs=[prompt_positive, prompt_negative],outputs=result)



demo.launch()