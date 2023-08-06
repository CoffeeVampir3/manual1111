import os, gc, random, sys, json, random, time
from mechanisms.mech_utils import get_path_from_leaf
from shared.scheduler_utils import get_scheduler_by_name
from mechanisms.pipe_utils import load_diffusers_pipe, get_rng_generator
from mechanisms.image_utils import save_images
from datetime import datetime
from mechanisms.tokenizers_utils import encode_from_pipe
from dataclasses import dataclass
from shared.config_utils import save_ui_config

def run_t2i(model_path, 
        positive_prompt, keyword_prompt, negative_prompt, negative_keyword_prompt,
        seed, classifier_free_guidance, generation_steps, batch_size, number_of_batches, scheduler_name):
    
    t2i_data = locals()
    save_ui_config(**t2i_data)
    
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
    
    pos, neg, pos_pool, neg_pool = encode_from_pipe(pipe, positive_prompt, negative_prompt, keyword_prompt, negative_keyword_prompt, batch_size)
    
    generation_configs = {
        "num_inference_steps":generation_steps,
        "width":1024,
        "height":1024,
        "guidance_scale":classifier_free_guidance,
    }
    
    all_images = []
    all_prompts = []
    for n in range(int(number_of_batches)):
        images = pipe(
            prompt_embeds = pos, 
            negative_prompt_embeds = neg, 
            pooled_prompt_embeds=pos_pool, 
            negative_pooled_prompt_embeds=neg_pool, 
            output_type = "pil", 
            generator=generator,
            **generation_configs).images
        all_images.extend(images)
        yield all_images
        del images
    
    current_time_as_text = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_images("outputs", current_time_as_text, all_images)
    #return all_images