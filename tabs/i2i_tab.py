import gradio as gr
import os
from tabs.tab_utils import get_available_from_dir, get_available_from_leafs
from mechanisms.run_pipe import run_i2i
from shared.config_utils import make_config_functions, get_component_dictionary
from functools import partial
from tabs.generator_components import make_prompt_column, make_generation_accordion

I2I_TAB_NAME = "image_to_image_v1"

def make_image_to_image_tab():
    with gr.Blocks() as interface:
        get_available_models = partial(get_available_from_leafs, "models", [".safetensors"])
        inputs = []
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    model_path = gr.Dropdown(choices = get_available_models(), label="Base model")
                conditioning, save_prompt, load_prompt = make_prompt_column(I2I_TAB_NAME)
                with gr.Accordion(label="Image"):
                    initialization_image = gr.Image(container=False, type="pil")
                    strength = gr.Slider(value=0.5, minimum=0.0, maximum=1.0, step=0.01, label="Strength")
                generating, submit, save_gen, load_gen = make_generation_accordion(I2I_TAB_NAME)

            with gr.Column(scale=3):
                output_gallery = gr.Gallery(
                    object_fit="contain", container=False,
                    preview=True, rows=2, height="90vh",
                    allow_preview=True)

    comp_dict = get_component_dictionary(locals())
    del comp_dict["initialization_image"]
    save_i2i, load_i2i, _ = make_config_functions(I2I_TAB_NAME, comp_dict, None)
    
    ui_items = [model_path, strength]
    
    inputs = [model_path, initialization_image, strength, *conditioning, *generating]
    interface.load(load_i2i, inputs=ui_items, outputs=ui_items)
    interface.load(load_prompt, inputs=conditioning, outputs=conditioning)
    interface.load(load_gen, inputs=generating, outputs=generating)
    
    submit.click(fn=save_i2i, inputs=ui_items, outputs=None)
    submit.click(fn=save_prompt, inputs=conditioning, outputs=None)
    submit.click(fn=save_gen, inputs=generating, outputs=None)
    submit.click(fn=run_i2i, inputs=inputs, outputs=output_gallery)
    
    return interface