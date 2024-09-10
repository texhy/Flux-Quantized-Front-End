import torch
from model import T5EncoderModel, FluxTransformer2DModel
from diffusers import FluxPipeline

def load_flux_pipeline():
    # Load models with appropriate dtype for CPU offloading
    text_encoder_2 = T5EncoderModel.from_pretrained(
        "HighCWu/FLUX.1-dev-4bit",
        subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16,
    )

    transformer = FluxTransformer2DModel.from_pretrained(
        "HighCWu/FLUX.1-dev-4bit",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    # Initialize the pipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        text_encoder_2=text_encoder_2,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    # Enable CPU offloading
    pipe.enable_model_cpu_offload()

    return pipe


def generate_image(pipe, prompt, height=1024, width=1024, guidance_scale=3.5, steps=16, seed=0):
    # Generate the image from the given prompt
    image = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        output_type="pil",
        num_inference_steps=steps,
        max_sequence_length=512,
        generator=torch.Generator().manual_seed(seed)  # Random seed generator
    ).images[0]

    return image
