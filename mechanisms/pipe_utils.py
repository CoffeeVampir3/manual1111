import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import os, gc, random, sys, json, random, time
from diffusers.utils.import_utils import is_xformers_available

global LOADED_PIPE
global LOADED_MODEL_PATH
LOADED_PIPE = None
LOADED_MODEL_PATH = None

def unload_current_pipe():
    global LOADED_PIPE
    global LOADED_MODEL_PATH
    
    if LOADED_PIPE:
        for x in LOADED_PIPE.__module__:
            del x
        del LOADED_PIPE
        
    LOADED_PIPE = None
    LOADED_MODEL_PATH = None

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
def load_diffusers_pipe(model_path, device):
    global LOADED_PIPE
    global LOADED_MODEL_PATH
    torch.set_float32_matmul_precision('high')
    
    if LOADED_PIPE and LOADED_MODEL_PATH and LOADED_MODEL_PATH == model_path:
        return LOADED_PIPE
    
    unload_current_pipe()
    
    #load new pipe
    try:
        LOADED_PIPE = StableDiffusionXLPipeline.from_single_file(
            model_path, 
            torch_dtype=torch.bfloat16, 
            use_safetensors=True, 
            variant="fp16", 
            add_watermarker=False)
        
        if is_xformers_available():
            try:
                LOADED_PIPE.enable_xformers_memory_efficient_attention()
            except:
                pass
        
        LOADED_PIPE.enable_vae_tiling()
        LOADED_PIPE.to(device)
        #if device != "cpu":
        #    LOADED_PIPE.enable_sequential_cpu_offload()
        LOADED_MODEL_PATH = model_path
    except Exception as e:
        print(f"Error loading the model: {e}")
        unload_current_pipe()
    
    return LOADED_PIPE

global GENERATOR
GENERATOR = None
def get_rng_generator(device):
    global GENERATOR
    GENERATOR = None
    if GENERATOR:
        return GENERATOR
    GENERATOR = torch.Generator(device)
    return GENERATOR
