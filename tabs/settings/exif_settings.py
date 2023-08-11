import gradio as gr
import os
from shared.running_config import set_config
from shared.config_utils import make_config_functions, get_component_dictionary, load_json_configs
from shared.log import logging

EXIF_TAB_NAME = "exif_settings_v1" 

def update_exif_config(**kwargs):
    if not kwargs:
        logging.debug("Loading from file...")
        kwargs = load_json_configs(EXIF_TAB_NAME)
    write_tags = kwargs["write_tags"]
    choices = []
    if "Write User Comment (Same as Automatic1111)" in write_tags: choices.append(37510)
    if "Write Image Description" in write_tags: choices.append(270)
    set_config("exif_tags", choices)
    logging.debug(f"Updated EXIF config {choices}")

def make_exif_settings():
    with gr.Blocks() as interface:
        with gr.Box():
            tag_options = ["Write User Comment (Same as Automatic1111)", "Write Image Description"]
            write_tags = gr.CheckboxGroup(label="Write EXIF Tag Types: ", 
                                          choices=tag_options, 
                                          container=False, 
                                          interactive=True, 
                                          value="Write Image Description", 
                                          type="value")
            
            ui_items = [write_tags]

    comp_dict = get_component_dictionary(locals())
    save, load, default = make_config_functions(EXIF_TAB_NAME, comp_dict, update_exif_config)
    
    interface.load(load, inputs=ui_items, outputs=ui_items)
    
    with gr.Row():
        save_button = gr.Button(value="Save Config")
        save_button.click(fn=save, inputs=ui_items, outputs=None)
        
        default_button = gr.Button(value="Return to default settings (Not saved until you hit save configs.)")
        default_button.click(fn=default, inputs=ui_items, outputs=ui_items)
    
    return interface