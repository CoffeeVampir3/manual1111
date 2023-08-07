import gradio as gr
from tabs.exif_settings import make_exif_settings

def make_config_tab():
    with gr.Blocks() as interface:
        with gr.Tab(label="Exif Settings"):
            make_exif_settings()
    return interface