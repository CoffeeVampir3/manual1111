import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import os, gc, random, sys, json, random, time
from diffusers.utils.import_utils import is_xformers_available

class NoWatermarker:
    def __init__(self):
        pass

    def apply_watermark(self, images: torch.FloatTensor):
        return images #Fake watermarker that bypasses watermarking.

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
    
def load_scheduler(pipe, scheduler):
    scheduler_dict = {
        "_diffusers_version": "0.19.0.dev0",
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "clip_sample": False,
        "interpolation_type": "linear",
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
        "sample_max_value": 1.0,
        "set_alpha_to_one": False,
        "skip_prk_steps": True,
        "steps_offset": 1,
        "timestep_spacing": "leading",
        "trained_betas": None,
        "use_karras_sigmas": False
    }

    pipe.scheduler = scheduler.from_config(scheduler_dict) 
    return pipe
    
def load_diffusers_pipe(model_path, scheduler, device):
    global LOADED_PIPE
    global LOADED_MODEL_PATH
    torch.set_float32_matmul_precision('high')
    
    #Check if there's already a loaded pipe that matches what kind of pipe we want.
    if (LOADED_MODEL_PATH and
        LOADED_MODEL_PATH == model_path and
        LOADED_PIPE):
        
        #Check if the scheduler is the correct one we want
        if ((not LOADED_PIPE.scheduler) or
            LOADED_PIPE.scheduler.__class__.__name__ != scheduler.__class__.__name__):
                load_scheduler(LOADED_PIPE, scheduler)
        
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
        LOADED_PIPE.watermark = NoWatermarker()
        
        LOADED_PIPE = load_scheduler(LOADED_PIPE, scheduler)
        
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
