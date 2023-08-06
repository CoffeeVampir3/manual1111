import json
from huggingface_hub import hf_hub_download

repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
index_file = hf_hub_download(repo_id=repo_id, filename="model_index.json")

with open(index_file, "r") as f:
    model_index = json.load(f)
    
for filename in model_index.keys():
    if filename.endswith(".fp16.safetensors"):
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir="models")