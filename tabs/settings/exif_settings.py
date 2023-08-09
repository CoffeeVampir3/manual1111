import gradio as gr
import os
from functools import partial
from shared.running_config import set_config
from shared.config_utils import save_ui_config, load_ui_config

EXIF_TAB_NAME = "exif_settings_v1"

def update_config(write_exif, write_tags):
    choices = []
    if "Write User Comment (Same as Automatic1111)" in write_tags: choices.append(37510)
    if "Write Image Description" in write_tags: choices.append(270)
    set_config("exif_tags", choices)

def update_tag_config(write_exif, write_tags):
    ui_state = locals()
    update_config(write_exif, write_tags)
    save_ui_config(EXIF_TAB_NAME, **ui_state)
        
def change_write_exif(choice):
    return gr.update(
        visible=choice, 
        value = "Write Image Description" if choice else None)
    
def make_exif_settings():
    with gr.Blocks() as interface:
        with gr.Box():
            write_exif = gr.Checkbox(label="Write Exif Data to Images", value=True, container=False)
            
            tag_options = ["Write User Comment (Same as Automatic1111)", "Write Image Description"]
            write_tags = gr.CheckboxGroup(choices=tag_options, container=False, visible=write_exif, interactive=True, value="Write Image Description", type="value")

            ui_items = [write_exif, write_tags]
    
    load_config = partial(load_ui_config, EXIF_TAB_NAME, ui_items)
    interface.load(load_config, inputs=None, outputs=ui_items)
    interface.load(update_config, inputs=ui_items, outputs=None)
    
    #Make sure to do this after load or else infinite loop.
    write_exif.change(fn=change_write_exif, inputs = write_exif, outputs=write_tags)
    write_tags.change(fn=update_tag_config, inputs = ui_items)
    
    return interface