import gradio as gr
from tabs.settings.exif_settings import make_exif_settings, update_exif_config
from tabs.settings.vae_settings import make_vae_settings, update_vae_config
from tabs.settings.optimization_settings import make_optimization_settings, update_optimization_config
from tabs.settings.scheduler_settings import make_scheduler_settings, load_scheduler_config

def load_all_configs():
     update_exif_config()
     update_optimization_config()
     update_vae_config()
     load_scheduler_config()

def make_config_tab():
    with gr.Blocks() as interface:
        with gr.Tab(label="Exif"):
            make_exif_settings()
        with gr.Tab(label="Vae"):
             make_vae_settings()
        with gr.Tab(label="Optimizations"):
             make_optimization_settings()
        with gr.Tab(label="Schedulers (Advanced)"):
             make_scheduler_settings()
    return interface