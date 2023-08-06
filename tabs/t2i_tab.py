import gradio as gr
import os
from tabs.tab_utils import get_available_from_dir, get_available_from_leafs
from mechanisms.t2i import run_t2i
from shared.scheduler_utils import get_available_scheduler_names
from shared.config_utils import load_ui_config
from functools import partial

def make_text_to_image_tab():
    with gr.Blocks() as interface:
        available_models = get_available_from_leafs("models")
        model_path = gr.Dropdown(choices = available_models, label="Base model")
        inputs = []
        with gr.Row():
            with gr.Column():
                with gr.Accordion(label="Conditioning", open=True):
                    with gr.Column():
                        with gr.Row():
                            positive_prompt = gr.TextArea(value="", lines=3, label="Positive Prompt")
                            keyword_prompt = gr.TextArea(value="", lines=3, label="Positive Keywords", visible=False)
                        with gr.Row():
                            negative_prompt = gr.TextArea(value="", lines=3, label="Negative Prompt")
                            negative_keyword_prompt = gr.TextArea(value="", lines=3, label="Negative Keywords", visible=False)
                    #conditioning = [positive_prompt, keyword_prompt, negative_prompt, negative_keyword_prompt]
                    conditioning = [positive_prompt, positive_prompt, negative_prompt, negative_prompt]
                
                with gr.Accordion(label="Config", open=True):
                    with gr.Row():
                        seed = gr.Number(value=int(-1), label="Seed")
                        classifier_free_guidance = gr.Slider(minimum=0.5, maximum=20.0, value=8.0, label="CFG")
                        generation_steps = gr.Slider(minimum=1.0, maximum=100.0, value=24.0, label="Steps")
                    with gr.Row():
                        batch_size = gr.Slider(minimum=1, maximum=20.0, value=4.0, step=1.0, label="# Per Run")
                        number_of_batches = gr.Slider(minimum=1, maximum=20.0, value=4.0, step=1.0, label="Run This Many Times")
                    scheduler_name = gr.Dropdown(choices = get_available_scheduler_names(), label="Scheduler")
                    generating = [seed, classifier_free_guidance, generation_steps, batch_size, number_of_batches, scheduler_name]

            with gr.Box():
                output_gallery = gr.Gallery()
                submit = gr.Button("Generate")
            inputs = [model_path, *conditioning, *generating]
            submit.click(fn=run_t2i, inputs=inputs, outputs=output_gallery)
    load_config = partial(load_ui_config, model_path)
    interface.load(load_config, inputs=None, outputs=inputs)
    return interface