import gradio as gr
import os
from shared.scheduler_utils import get_available_scheduler_names
from mechanisms.killswitch import killswitch_engage
from shared.config_utils import make_config_functions, get_component_dictionary

def make_prompt_column(tab_name):
    column_postfix = "-prompt"
    config_name = tab_name + column_postfix
    with gr.Column():
        with gr.Group():
            with gr.Row():
                positive_prompt = gr.TextArea(value="", lines=3, label="Positive Prompt", container=True)
            with gr.Row():
                negative_prompt = gr.TextArea(value="", lines=3, label="Negative Prompt", container=True)

    comp_dict = get_component_dictionary(locals())
    save, load, _ = make_config_functions(config_name, comp_dict, None)
    
    return [positive_prompt, negative_prompt], save, load

def make_generation_accordion(tab_name):
    column_postfix = "-generation"
    config_name = tab_name + column_postfix
    with gr.Accordion(label="Config", open=True):
        with gr.Group():
            with gr.Row():
                seed = gr.Number(value=int(-1), label="Seed", precision=0)
                classifier_free_guidance = gr.Slider(minimum=0.5, maximum=200.0, value=16.0, label="CFG")
                generation_steps = gr.Slider(minimum=1, maximum=100, step=int(1), value=int(24), label="Steps")
            with gr.Row():
                image_width = gr.Slider(minimum=64, maximum=2048, step=int(8), value=int(1024), label="Width")
                image_height = gr.Slider(minimum=64, maximum=2048, step=int(8), value=int(1024), label="Height")
            with gr.Row():
                batch_size = gr.Slider(minimum=1, maximum=20, value=int(1), step=int(1), label="# Per Run")
                number_of_batches = gr.Slider(minimum=1, maximum=20, value=int(1), step=int(1), label="Run This Many Times")
            scheduler_name = gr.Dropdown(choices = get_available_scheduler_names(), label="Scheduler", value = "HeunDiscrete")
        with gr.Column():
            with gr.Row():
                kill = gr.Button("Stop")
                kill.click(fn=killswitch_engage, queue=False)
                submit = gr.Button("Generate")
                
    comp_dict = get_component_dictionary(locals())
    save, load, _ = make_config_functions(config_name, comp_dict, None)
    
    return [seed, classifier_free_guidance, generation_steps, image_width, image_height, batch_size, number_of_batches, scheduler_name], submit, save, load