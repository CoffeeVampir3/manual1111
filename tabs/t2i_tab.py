import gradio as gr
import os
from tabs.tab_utils import get_available_from_dir, get_available_from_leafs
from mechanisms.t2i import run_t2i, T2I_TAB_NAME
from shared.scheduler_utils import get_available_scheduler_names
from shared.config_utils import load_json_configs
from functools import partial
from mechanisms.killswitch import killswitch_engage

def hacked_load(model_path, 
        positive_prompt, keyword_prompt, negative_prompt, negative_keyword_prompt,
        seed, classifier_free_guidance, generation_steps, image_width, image_height,
        batch_size, number_of_batches, scheduler_name):
    ui_state = locals()
    return load_json_configs(T2I_TAB_NAME, **ui_state)

def make_text_to_image_tab():
    with gr.Blocks() as interface:
        get_available_models = partial(get_available_from_leafs, "models", [".safetensors"])
        inputs = []
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    model_path = gr.Dropdown(choices = get_available_models(), label="Base model")
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            positive_prompt = gr.TextArea(value="", lines=3, label="Positive Prompt", container=True)
                            keyword_prompt = gr.TextArea(value="", lines=3, label="Positive Keywords", container=True, visible=False) #Unused for now.
                        with gr.Row():
                            negative_prompt = gr.TextArea(value="", lines=3, label="Negative Prompt", container=True)
                            negative_keyword_prompt = gr.TextArea(value="", lines=3, label="Negative Keywords", container=True, visible=False) #Unused for now.
                        #conditioning = [positive_prompt, keyword_prompt, negative_prompt, negative_keyword_prompt]
                        conditioning = [positive_prompt, positive_prompt, negative_prompt, negative_prompt]
                
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
                        generating = [seed, classifier_free_guidance, generation_steps, image_width, image_height, batch_size, number_of_batches, scheduler_name]
                with gr.Column():
                    with gr.Row():
                        kill = gr.Button("Stop")
                        kill.click(fn=killswitch_engage, queue=False)
                        submit = gr.Button("Generate")

            with gr.Column(scale=3):
                output_gallery = gr.Gallery(
                    object_fit="contain", container=False, 
                    preview=True, rows=2, height="90vh", 
                    allow_preview=True)

    inputs = [model_path, *conditioning, *generating]
    interface.load(hacked_load, inputs=inputs, outputs=inputs)
    submit.click(fn=run_t2i, inputs=inputs, outputs=output_gallery)
    
    return interface