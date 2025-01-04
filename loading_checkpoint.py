import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
import numpy as np

# Function to load the UNet model from a checkpoint
def load_unet_from_checkpoint(checkpoint_dir):
    unet = UNet2DConditionModel.from_pretrained(checkpoint_dir, subfolder="unet")
    return unet

def load_checkpoint(checkpoint_dir):
    # Download Stable Diffusion 1.4 pipeline
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    
    # Load the custom UNet model from the checkpoint
    custom_unet = load_unet_from_checkpoint(checkpoint_dir)
    
    # Replace the UNet in the pipeline with the custom UNet
    pipeline.unet = custom_unet
    
    return pipeline