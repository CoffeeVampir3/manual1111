import gradio as gr
import os
from shared.running_config import set_config
from shared.config_utils import save_json_configs, load_json_configs

VAE_TAB_NAME = "vae_settings_v1"

def update_config(use_fast_decoder):
    set_config("use_fast_decoder", use_fast_decoder)

def update_and_save(use_fast_decoder):
    ui_state = locals()
    update_config(use_fast_decoder)
    save_json_configs(VAE_TAB_NAME, **ui_state)

def hacked_load(use_fast_decoder):
    ui_state = locals()
    vals = load_json_configs(VAE_TAB_NAME, **ui_state)
    return vals[0] #[0] Super important to not cause an infinite loop for... reasons
    
def make_vae_settings():        
    with gr.Blocks() as interface:
        with gr.Box():
            use_fast_decoder = gr.Checkbox(label="Use TAESDXL fast decoder for final decoding, minor quality degradation", value=False, container=False)
            ui_items = [use_fast_decoder]
    
    interface.load(hacked_load, inputs=ui_items, outputs=ui_items)
    
    #Make sure to do this after load or else infinite loop.
    use_fast_decoder.change(fn=update_and_save, inputs = ui_items)

    return interface