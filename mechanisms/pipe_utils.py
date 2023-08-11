import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import os, gc, random, sys, json, random, time
from diffusers.utils.import_utils import is_xformers_available
from shared.running_config import get_config
from shared.log import vampire_log
from shared.scheduler_utils import get_name_by_scheduler
from functools import partial

class NoWatermarker:
    def __init__(self):
        pass

    def apply_watermark(self, images: torch.FloatTensor):
        return images #Fake watermarker that bypasses watermarking.

global LOADED_PIPE
global LOADED_MODEL_PATH
global CURRENT_COMPILATION_METHOD
LOADED_PIPE = None
LOADED_MODEL_PATH = None
CURRENT_COMPILATION_METHOD = None

def unload_current_pipe():
    global LOADED_PIPE
    global LOADED_MODEL_PATH
    global CURRENT_COMPILATION_METHOD
    
    if LOADED_PIPE:
        for x in LOADED_PIPE.__module__:
            del x
        del LOADED_PIPE
        
    LOADED_PIPE = None
    LOADED_MODEL_PATH = None
    CURRENT_COMPILATION_METHOD = None

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
def load_new_pipe(model_path, scheduler, device):
    global LOADED_PIPE
    global LOADED_MODEL_PATH
    unload_current_pipe()
    
    from shared.running_config import debug_config
    try:
        load_pipe_func = StableDiffusionXLPipeline.from_single_file
        
        scheduler_name = get_name_by_scheduler(scheduler)
        scheduler_config = get_config(scheduler_name)
        if scheduler_config: load_pipe_func = partial(load_pipe_func, scheduler_config=scheduler_config)
        
        LOADED_PIPE = load_pipe_func(
            model_path, 
            torch_dtype=torch.bfloat16, 
            use_safetensors=True, 
            variant="fp16",
            add_watermarker=False)
        
        LOADED_MODEL_PATH = model_path
        LOADED_PIPE.watermark = NoWatermarker()
        
        LOADED_PIPE = load_scheduler(LOADED_PIPE, scheduler)
        LOADED_PIPE = load_vae(LOADED_PIPE, model_path, scheduler, device)
        LOADED_PIPE = load_compiler(LOADED_PIPE, model_path, scheduler, device)

        LOADED_PIPE.to(device)
        #if device != "cpu":
        #    LOADED_PIPE.enable_sequential_cpu_offload()
        return LOADED_PIPE
 
    except Exception as e:
        vampire_log.critical(f"Error loading the model {e}")
        unload_current_pipe()
    
    return LOADED_PIPE

def load_compiler(pipe, model_path, scheduler, device):
    global CURRENT_COMPILATION_METHOD
    compilation_method = get_config("compilation_method")
    optimization_type = get_config("optimize_for")
    
    if CURRENT_COMPILATION_METHOD and CURRENT_COMPILATION_METHOD == compilation_method:
        return pipe
    
    if CURRENT_COMPILATION_METHOD != None:
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
    
    CURRENT_COMPILATION_METHOD = compilation_method
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

    
def load_diffusers_pipe(model_path, scheduler, device):
    global LOADED_PIPE
    global LOADED_MODEL_PATH
    torch.set_float32_matmul_precision('high')
    
    #Check if there's already a loaded pipe that matches what kind of pipe we want.
    if (LOADED_MODEL_PATH and
        LOADED_MODEL_PATH == model_path and
        LOADED_PIPE):
        
        LOADED_PIPE = load_scheduler(LOADED_PIPE, scheduler)
        LOADED_PIPE = load_vae(LOADED_PIPE, model_path, scheduler, device)
        LOADED_PIPE = load_compiler(LOADED_PIPE, model_path, scheduler, device)
        return LOADED_PIPE
    
    return load_new_pipe(model_path, scheduler, device)

global GENERATOR
GENERATOR = None
def get_rng_generator(device):
    global GENERATOR
    GENERATOR = None
    if GENERATOR:
        return GENERATOR
    GENERATOR = torch.Generator(device)
    return GENERATOR
