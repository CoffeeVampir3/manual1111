import gradio as gr
import os
from shared.running_config import set_config
from shared.config_utils import save_json_configs, load_json_configs

OPTIMIZATION_TAB_NAME = "optimization_settings_v1"

def update_config(compilation_method, optimize_for):
    if compilation_method == "None": compilation_method = None
    set_config("compilation_method", compilation_method)
    
    optim_target = None
    if optimize_for == "Memory Efficiency":                         optim_target = "reduce-overhead"
    if optimize_for == "Speed":                                     optim_target = "default"
    if optimize_for == "Maximized (Potentially Very slow startup)": optim_target = "max-autotune"
    set_config("optimize_for", optim_target)

def update_and_save(compilation_method, optimize_for):
    ui_state = locals()
    update_config(compilation_method, optimize_for)
    save_json_configs(OPTIMIZATION_TAB_NAME, **ui_state)

def hacked_load(compilation_method, optimize_for):
    ui_state = locals()
    return load_json_configs(OPTIMIZATION_TAB_NAME, **ui_state)

def make_optimization_settings():
    with gr.Blocks() as interface:
        with gr.Box():
            gr.Label(value="""Use a torch compiler to dramatically increase generation speed. There's some caveats though, changing some parameters might trigger a re-compilation.
                     This is an experimental torch feature, so be warned that you might have issues when using this.""")
            with gr.Row():
                #inductor seems to be bugged, need to debug
                compilation_method = gr.Dropdown(choices=["None", "inductor", "eager", "aot_eager"], value="None", label="Use Compilation Method")
                optimize_for = gr.Dropdown(choices=["Memory Efficiency", "Speed", "Maximized (Potentially Very slow startup)"], value="Speed", label="Optimize For (Inductor only)")
        
            ui_items = [compilation_method, optimize_for]
    
    interface.load(hacked_load, inputs=ui_items, outputs=ui_items)
    
    compilation_method.change(fn=update_and_save, inputs = ui_items)
    optimize_for.change(fn=update_and_save, inputs = ui_items)
    
    return interface