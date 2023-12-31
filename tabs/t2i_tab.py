import gradio as gr
import os
from tabs.tab_utils import get_available_from_dir, get_available_from_leafs
from mechanisms.run_pipe import run_t2i
from shared.config_utils import make_config_functions, get_component_dictionary
from functools import partial
from tabs.generator_components import make_prompt_column, make_generation_accordion

T2I_TAB_NAME = "text_to_image_v1"

def make_text_to_image_tab():
    with gr.Blocks() as interface:
        get_available_models = partial(get_available_from_leafs, "models", [".safetensors"])
        inputs = []
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    model_path = gr.Dropdown(choices = get_available_models(), label="Base model")
                conditioning, save_prompt, load_prompt = make_prompt_column(T2I_TAB_NAME)
                
                with gr.Row():
                    image_width = gr.Slider(minimum=64, maximum=2048, step=int(8), value=int(1024), label="Width")
                    image_height = gr.Slider(minimum=64, maximum=2048, step=int(8), value=int(1024), label="Height")
                
                generating, submit, save_gen, load_gen = make_generation_accordion(T2I_TAB_NAME)

            with gr.Column(scale=3):
                output_gallery = gr.Gallery(
                    object_fit="contain", container=False, 
                    preview=True, rows=2, height="90vh", 
                    allow_preview=True)

    comp_dict = get_component_dictionary(locals())
    save_t2i, load_t2i, _ = make_config_functions(T2I_TAB_NAME, comp_dict, None)
    
    ui_items = [model_path, image_width, image_height]
    
    inputs = [*ui_items, *conditioning, *generating]
    interface.load(load_t2i, inputs=ui_items, outputs=ui_items)
    interface.load(load_prompt, inputs=conditioning, outputs=conditioning)
    interface.load(load_gen, inputs=generating, outputs=generating)
    
    submit.click(fn=save_t2i, inputs=ui_items, outputs=None)
    submit.click(fn=save_prompt, inputs=conditioning, outputs=None)
    submit.click(fn=save_gen, inputs=generating, outputs=None)
    
    submit.click(fn=run_t2i, inputs=inputs, outputs=output_gallery)
    
    return interface