import gradio as gr
from tabs.settings.exif_settings import make_exif_settings
from tabs.settings.vae_settings import make_vae_settings

def make_config_tab():
    with gr.Blocks() as interface:
        with gr.Tab(label="Vae Settings"):
            make_vae_settings()
        with gr.Tab(label="Exif Settings"):
            make_exif_settings()
    return interface