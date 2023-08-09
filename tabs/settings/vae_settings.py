import gradio as gr
import os
from functools import partial
from shared.running_config import set_config
from shared.config_utils import save_ui_config, load_ui_config

VAE_TAB_NAME = "vae_settings_v1"

def update_config(use_fast_decoder):
    set_config("use_fast_decoder", use_fast_decoder)

def update_and_save(use_fast_decoder):
    ui_state = locals()
    update_config(use_fast_decoder)
    save_ui_config(VAE_TAB_NAME, **ui_state)
    
def initialize(use_fast_decoder):
    load_config = partial(load_ui_config, VAE_TAB_NAME, [use_fast_decoder])
    conf = load_config()
    return gr.update(value=conf[0])
    
def make_vae_settings():        
    with gr.Blocks() as interface:
        with gr.Box():
            use_fast_decoder = gr.Checkbox(label="Use TAESDXL fast decoder for final decoding, minor quality degradation", value=False, container=False, interactive=True)
            ui_items = [use_fast_decoder]
            
    load_config = partial(load_ui_config, VAE_TAB_NAME, ui_items)
    conf = load_config()
    interface.load(initialize, inputs=use_fast_decoder, outputs=use_fast_decoder)
    
    #Make sure to do this after load or else infinite loop.
    use_fast_decoder.change(fn=update_and_save, inputs = ui_items)

    return interface