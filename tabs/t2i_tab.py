import gradio as gr
import os
from tabs.tab_utils import get_available_from_dir, get_available_from_leafs
from mechanisms.t2i import run_t2i
from shared.scheduler_utils import get_available_scheduler_names

def make_text_to_image_tab():
    with gr.Blocks() as interface:
        available_models = get_available_from_leafs("models")
        model_path = gr.Dropdown(choices = available_models, label="Base model")
        with gr.Row():
            with gr.Column():
                with gr.Accordion(label="Conditioning", open=True):
                    with gr.Column():
                        with gr.Row():
                            positive_prompt = gr.TextArea(value="", lines=3, label="Positive Prompt")
                            keyword_prompt = gr.TextArea(value="", lines=3, label="Positive Keywords")
                        with gr.Row():
                            negative_prompt = gr.TextArea(value="", lines=3, label="Negative Prompt")
                            negative_keyword_prompt = gr.TextArea(value="", lines=3, label="Negative Keywords")
                    conditioning = [positive_prompt, keyword_prompt, negative_prompt, negative_keyword_prompt]
                
                with gr.Accordion(label="Config", open=True):
                    with gr.Row():
                        seed = gr.Number(value=int(-1), label="Generation Seed")
                        classifier_free_guidance = gr.Slider(minimum=0.5, maximum=20.0, value=8.0, label="Classifier Free Guidance")
                        batch_size = gr.Slider(minimum=1, maximum=20.0, value=4.0, step=1.0, label="Batch Size")
                    scheduler_name = gr.Dropdown(choices = get_available_scheduler_names(), label="Scheduler")
                    generating = [seed, classifier_free_guidance, batch_size, scheduler_name]

            with gr.Box():
                output_gallery = gr.Gallery()
                submit = gr.Button("Generate")

            submit.click(fn=run_t2i, inputs=[model_path, *conditioning, *generating], outputs=output_gallery)
    return interface