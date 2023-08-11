import gradio as gr
import os
from shared.running_config import set_config
from shared.config_utils import get_config_save_load, get_component_dictionary

EXIF_TAB_NAME = "exif_settings_v1" 

def update_config(**kwargs):
    write_tags = kwargs["write_tags"]
    choices = []
    if "Write User Comment (Same as Automatic1111)" in write_tags: choices.append(37510)
    if "Write Image Description" in write_tags: choices.append(270)
    set_config("exif_tags", choices)
    print("Updated EXIF config.")

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

    comp_dict = get_component_dictionary(locals())
    save, load = get_config_save_load(EXIF_TAB_NAME, comp_dict, update_config)
    
    interface.load(load, inputs=write_tags, outputs=write_tags)
    
    save_button = gr.Button(value="Save Config")
    save_button.click(fn=save, inputs=write_tags, outputs=None)
    
    return interface