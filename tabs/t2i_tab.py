import gradio as gr
import os
from tabs.tab_utils import get_available_from_dir, get_available_from_leafs
from mechanisms.t2i import run_t2i
from shared.scheduler_utils import get_available_scheduler_names
from shared.config_utils import load_ui_config
from functools import partial
from mechanisms.killswitch import killswitch_engage

def make_text_to_image_tab():
    with gr.Blocks() as interface:
        available_models = get_available_from_leafs("models")
        inputs = []
        with gr.Row():
            with gr.Column(scale=2):
                model_path = gr.Dropdown(choices = available_models, label="Base model")
                with gr.Accordion(label="Conditioning", open=True):
                    with gr.Column():
                        with gr.Group():
                            with gr.Row():
                                positive_prompt = gr.TextArea(value="", lines=3, label="Positive Prompt")
                                keyword_prompt = gr.TextArea(value="", lines=3, label="Positive Keywords", visible=False) #Unused for now.
                            with gr.Row():
                                negative_prompt = gr.TextArea(value="", lines=3, label="Negative Prompt")
                                negative_keyword_prompt = gr.TextArea(value="", lines=3, label="Negative Keywords", visible=False) #Unused for now.
                            #conditioning = [positive_prompt, keyword_prompt, negative_prompt, negative_keyword_prompt]
                            conditioning = [positive_prompt, positive_prompt, negative_prompt, negative_prompt]
                
                with gr.Accordion(label="Config", open=True):
                    with gr.Group():
                        with gr.Row():
                            seed = gr.Number(value=int(-1), label="Seed")
                            classifier_free_guidance = gr.Slider(minimum=0.5, maximum=20.0, value=8.0, label="CFG")
                            generation_steps = gr.Slider(minimum=1, maximum=100, step=int(1), value=int(24), label="Steps")
                        with gr.Row():
                            batch_size = gr.Slider(minimum=1, maximum=20, value=int(1), step=int(1), label="# Per Run")
                            number_of_batches = gr.Slider(minimum=1, maximum=20, value=int(1), step=int(1), label="Run This Many Times")
                        scheduler_name = gr.Dropdown(choices = get_available_scheduler_names(), label="Scheduler")
                        generating = [seed, classifier_free_guidance, generation_steps, batch_size, number_of_batches, scheduler_name]
                with gr.Column():
                    with gr.Row():
                        kill = gr.Button("Stop")
                        kill.click(fn=killswitch_engage, queue=False)
                        submit = gr.Button("Generate")

            with gr.Column(scale=3):
                output_gallery = gr.Gallery(
                    object_fit="contain", container=False, 
                    preview=True, rows=2, height="85vh", 
                    allow_preview=True)

    inputs = [model_path, *conditioning, *generating]
    load_config = partial(load_ui_config, "text_to_image_v1", model_path)
    submit.click(fn=run_t2i, inputs=inputs, outputs=output_gallery)
    interface.load(load_config, inputs=None, outputs=inputs)
    return interface