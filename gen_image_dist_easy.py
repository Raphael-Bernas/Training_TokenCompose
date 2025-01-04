import torch
import json
import argparse
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from accelerate import PartialState
from loading_checkpoint import load_checkpoint

parser = argparse.ArgumentParser()

# python gen_image_dist_easy.py --output_dir sample_imgs/TokenCompose_SD14_A --model_name mlpc-lab/TokenCompose_SD14_A 
# --img_per_prompt 10 --resolution 256 --num_inference_steps 10 --num_images 100

parser.add_argument("--text_file_path", type=str, default="coco_obj_comp_5_1k.json")
parser.add_argument("--output_dir", type=str, default="sample_imgs/StableDiffusion1.4")
parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--img_per_prompt", type=int, default=2)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--num_inference_steps", type=int, default=10)
parser.add_argument("--num_images", type=int, default=100)

args = parser.parse_args()

model_name = args.model_name
if "checkpoint" in model_name:
    pipe = load_checkpoint(model_name, torch_dtype=torch.float32)
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)

distributed_state = PartialState()
pipe.to(distributed_state.device)
pipe.set_progress_bar_config(disable=True)

def dummy_checker(images, **kwargs):
    return images, [False] * len(images)

pipe.safety_checker = dummy_checker
print("Model loaded.")

text_file_path = args.text_file_path

with open(text_file_path, "r") as f:
    text_data = json.load(f)

# prepare the data
d = []

num_imgs = min(args.num_images, len(text_data))
for index in range(num_imgs):
    texts = [text_data[index]["text"] for _ in range(args.img_per_prompt)]
    for j in range(args.img_per_prompt):
        d.append((index, texts[j], j))
print("Data prepared.")

print("Start generating images.")
with distributed_state.split_between_processes(d) as data:
    for index, text, j in tqdm(data):
        img_id = "{}_{}".format(index, j)
        save_path = f"{args.output_dir}/{img_id}.png"
        image = pipe(prompt=[text], height=args.resolution, width=args.resolution, num_inference_steps=args.num_inference_steps).images[0]
        image.save(save_path)
