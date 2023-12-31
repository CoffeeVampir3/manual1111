import gradio as gr
import os
from shared.running_config import set_config
from shared.config_utils import make_config_functions, get_component_dictionary, load_json_configs
from shared.log import vampire_log

VAE_TAB_NAME = "vae_settings_v1"

def update_vae_config(**kwargs):
    if not kwargs:
        vampire_log.debug("Loading from file...")
        kwargs = load_json_configs(VAE_TAB_NAME)
    use_fast_decoder = kwargs["use_fast_decoder"]
    set_config("use_fast_decoder", use_fast_decoder)
    vampire_log.debug(f"Updated VAE Config {use_fast_decoder}")
    
def make_vae_settings():        
    with gr.Blocks() as interface:
        with gr.Box():
            use_fast_decoder = gr.Checkbox(label="Use TAESDXL fast decoder for final decoding, minor quality degradation", value=False, container=False)
            ui_items = [use_fast_decoder]
    
    comp_dict = get_component_dictionary(locals())
    save, load, default = make_config_functions(VAE_TAB_NAME, comp_dict, update_vae_config)
    
    interface.load(load, inputs=ui_items, outputs=ui_items)
    
    with gr.Row():
        save_button = gr.Button(value="Save Config")
        save_button.click(fn=save, inputs=ui_items, outputs=None)
        
        default_button = gr.Button(value="Return to default settings (Not saved until you hit save configs.)")
        default_button.click(fn=default, inputs=ui_items, outputs=ui_items)

    return interface