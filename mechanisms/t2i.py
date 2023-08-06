import os, gc, random, sys, json, random, time
from mechanisms.mech_utils import get_path_from_leaf
from shared.scheduler_utils import get_scheduler_by_name
from mechanisms.pipe_utils import load_diffusers_pipe, get_rng_generator
from mechanisms.image_utils import save_images
from datetime import datetime
from mechanisms.tokenizers_utils import encode_from_pipe

def run_t2i(model_path, 
            positive_prompt, positive_keywords, negative_prompt, negative_keywords, 
            seed, cfg, batch_size, scheduler_name):
    scheduler = get_scheduler_by_name(scheduler_name)
    resolved_model_path = get_path_from_leaf("models", model_path)
    device = "cuda"
    batch_size = int(batch_size)
    
    pipe = load_diffusers_pipe(resolved_model_path, device)
    
    ##seed generator
    generator = get_rng_generator(device)
    if seed == -1: nseed = random.randint(0, (sys.maxsize/64)) #random seed
    else: nseed = seed
    generator.manual_seed(nseed)
    
    pos, neg, pos_pool, neg_pool = encode_from_pipe(pipe, positive_prompt, negative_prompt, positive_keywords, negative_keywords, batch_size)
    
    generation_configs = {
        "num_inference_steps":35,
        "width":1024,
        "height":1024,
        "guidance_scale":cfg,
        #"guidance_rescale":0.7,
    }
    
    images = pipe(
        prompt_embeds = pos, 
        negative_prompt_embeds = neg, 
        pooled_prompt_embeds=pos_pool, 
        negative_pooled_prompt_embeds=neg_pool, 
        output_type = "pil", 
        generator=generator,
        **generation_configs).images
    
    current_time_as_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_images("outputs", current_time_as_text, images, positive_prompt)
    return images