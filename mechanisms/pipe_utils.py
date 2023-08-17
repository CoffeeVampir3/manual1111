import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import os, gc, random, sys, json, random, time
from diffusers.utils.import_utils import is_xformers_available
from shared.running_config import get_config
from shared.log import vampire_log
from shared.scheduler_utils import get_name_by_scheduler
from functools import partial
from dataclasses import dataclass

class NoWatermarker:
    def __init__(self):
        pass

    def apply_watermark(self, images: torch.FloatTensor):
        return images #Fake watermarker that bypasses watermarking.

@dataclass
class CurrentPipeConfig:
    pipe: None
    path: None
    compiler: None
    pipe_type: None
    
def reset_pipe_config():
    global CURRENT_PIPE_CONFIG 
    CURRENT_PIPE_CONFIG = CurrentPipeConfig(None, None, None, None)

reset_pipe_config()

def unload_current_pipe():
    global CURRENT_PIPE_CONFIG
    
    if CURRENT_PIPE_CONFIG.pipe:
        for x in CURRENT_PIPE_CONFIG.pipe.__module__:
            del x
        del CURRENT_PIPE_CONFIG.pipe
        
    reset_pipe_config()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
def load_new_pipe(model_path, scheduler, device, load_pipe_func):
    global CURRENT_PIPE_CONFIG
    unload_current_pipe()
    
    from shared.running_config import debug_config
    try:     
        scheduler_name = get_name_by_scheduler(scheduler)
        scheduler_config = get_config(scheduler_name)
        if scheduler_config: load_pipe_func = partial(load_pipe_func, scheduler_config=scheduler_config)
        
        pipe = load_pipe_func(
            model_path, 
            torch_dtype=torch.bfloat16, 
            use_safetensors=True, 
            variant="fp16",
            add_watermarker=False)
        
        CURRENT_PIPE_CONFIG.path = model_path
        pipe.watermark = NoWatermarker()
        
        pipe = load_scheduler(pipe, scheduler)
        pipe = load_vae(pipe, model_path, scheduler, device)
        pipe = load_compiler(pipe, model_path, scheduler, device)

        pipe.to(device)
        if device != "cpu" and get_config("use_cpu_offloading"):
            pipe.enable_sequential_cpu_offload()
            
        CURRENT_PIPE_CONFIG.pipe = pipe
        return CURRENT_PIPE_CONFIG.pipe
 
    except Exception as e:
        vampire_log.critical(f"Error loading the model {e}")
        unload_current_pipe()
    
    return CURRENT_PIPE_CONFIG.pipe

def load_compiler(pipe, model_path, scheduler, device):
    global CURRENT_PIPE_CONFIG
    compilation_method = get_config("compilation_method")
    optimization_type = get_config("optimize_for")
    current_method = CURRENT_PIPE_CONFIG.compiler
    
    if current_method and current_method == compilation_method:
        return pipe
    
    if current_method != None:
        #Safe because unload_current_pipe resets the current compilation method after unloading.
        return load_new_pipe(model_path, scheduler, device)
    
    if compilation_method:
        compile_func = partial(torch.compile, backend=compilation_method)
        if compilation_method == "inductor":
            compile_func = partial(compile_func, mode=optimization_type)
        else:
            optimization_type = None
        pipe.unet = compile_func(pipe.unet)
        vampire_log.warn(f"Compiler set: {compilation_method} {optimization_type}")
        
    elif is_xformers_available():
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    
    CURRENT_PIPE_CONFIG.compiler = compilation_method
    return pipe
    
def load_scheduler(pipe, scheduler):
    if (pipe.scheduler and 
        pipe.scheduler.__class__.__name__ == scheduler.__name__):
            return pipe
    
    vampire_log.warn("Loading a new scheduler.")
    scheduler_name = get_name_by_scheduler(scheduler)
    scheduler_config = get_config(scheduler_name)
    
    if not scheduler_config:
        vampire_log.warn("No scheduler config found, using default.")
        scheduler_config = {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "interpolation_type": "linear",
            "prediction_type": "epsilon",
            "sample_max_value": 1.0,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "timestep_spacing": "leading",
            "trained_betas": None,
            "use_karras_sigmas": False
        }

    pipe.scheduler = scheduler.from_config(scheduler_config) 
    return pipe
 
def load_vae(pipe, model_path, scheduler, device):
    if get_config("use_fast_decoder") and pipe.vae and pipe.vae.__class__.__name__ != "AutoencoderTiny":
        from diffusers import AutoencoderTiny
        try:
            vampire_log.debug("Loaded TAESDXL VAe.")
            pipe.vae = AutoencoderTiny.from_pretrained("vaes/taesdxl", torch_dtype=torch.bfloat16)
            return pipe
        except:
            vampire_log.critical("Wasn't able to load TAESDXL fast decoder. Run the install.py script to download it! Forcing config to normal VAE, falling back to normal VAE...")
            from shared.running_config import set_config
            set_config("use_fast_decoder", False)
        
    if not pipe.vae:
        #This is not an infinite loop because we force the config to use the normal vae in case it failed to load.
        pipe = load_new_pipe(model_path, scheduler, device)
    
    if pipe.vae.__class__.__name__ != "AutoencoderTiny":
        pipe.enable_vae_tiling()

    return pipe
        
def load_diffusers_pipe(model_path, scheduler, device, pipe_type, load_pipe_func):
    torch.set_float32_matmul_precision('high')
    
    global CURRENT_PIPE_CONFIG
    
    new_pipe_config = CurrentPipeConfig(CURRENT_PIPE_CONFIG.pipe, model_path, CURRENT_PIPE_CONFIG.compiler, pipe_type.__name__)
    
    #Check if there's already a loaded pipe that matches what kind of pipe we want.
    if (CURRENT_PIPE_CONFIG.pipe and CURRENT_PIPE_CONFIG == new_pipe_config):
        pipe = CURRENT_PIPE_CONFIG.pipe
        pipe = load_scheduler(pipe, scheduler)
        pipe = load_vae(pipe, model_path, scheduler, device)
        pipe = load_compiler(pipe, model_path, scheduler, device)
        CURRENT_PIPE_CONFIG.pipe = pipe
        return CURRENT_PIPE_CONFIG.pipe
    
    CURRENT_PIPE_CONFIG.pipe_type = pipe_type.__name__
    return load_new_pipe(model_path, scheduler, device, load_pipe_func)

global GENERATOR
GENERATOR = None
def get_rng_generator(device):
    global GENERATOR
    GENERATOR = None
    if GENERATOR:
        return GENERATOR
    GENERATOR = torch.Generator(device)
    return GENERATOR
