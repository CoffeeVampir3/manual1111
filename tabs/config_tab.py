import gradio as gr
from tabs.settings.exif_settings import make_exif_settings
from tabs.settings.vae_settings import make_vae_settings
from tabs.settings.optimization_settings import make_optimization_settings

def make_config_tab():
    with gr.Blocks() as interface:
        with gr.Tab(label="Vae"):
            make_vae_settings()
        with gr.Tab(label="Exif"):
            make_exif_settings()
        with gr.Tab(label="Optimizations"):
            make_optimization_settings()
    return interface