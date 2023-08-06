import gradio as gr
from tabs.t2i_tab import make_text_to_image_tab

with gr.Blocks() as interface:
    make_text_to_image_tab()
interface.launch(enable_queue=True)