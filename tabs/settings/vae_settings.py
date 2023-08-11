import gradio as gr
import os
from shared.running_config import set_config
from shared.config_utils import get_config_save_load, get_component_dictionary

VAE_TAB_NAME = "vae_settings_v1"

def update_config(**kwargs):
    use_fast_decoder = kwargs["use_fast_decoder"]
    set_config("use_fast_decoder", use_fast_decoder)
    print("Updated VAE Config")
    
def make_vae_settings():        
    with gr.Blocks() as interface:
        with gr.Box():
            use_fast_decoder = gr.Checkbox(label="Use TAESDXL fast decoder for final decoding, minor quality degradation", value=False, container=False)
            ui_items = [use_fast_decoder]
    
    comp_dict = get_component_dictionary(locals())
    save, load = get_config_save_load(VAE_TAB_NAME, comp_dict, update_config)
    
    interface.load(load, inputs=ui_items, outputs=ui_items)
    
    save_button = gr.Button(value="Save Config")
    save_button.click(fn=save, inputs=ui_items, outputs=None)

    return interface