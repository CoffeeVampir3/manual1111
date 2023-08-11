import gradio as gr
import os
from shared.running_config import set_config
from shared.config_utils import make_config_functions, get_component_dictionary, load_json_configs
from shared.log import vampire_log

OPTIMIZATION_TAB_NAME = "optimization_settings_v1"

def update_optimiazation_config(**kwargs):
    if not kwargs:
        vampire_log.debug("Loading from file...")
        kwargs = load_json_configs(OPTIMIZATION_TAB_NAME)
    compilation_method, optimize_for = kwargs["compilation_method"], kwargs["optimize_for"]
    if compilation_method == "None": compilation_method = None
    set_config("compilation_method", compilation_method)
    
    optim_target = None
    if optimize_for == "Memory Efficiency":                         optim_target = "reduce-overhead"
    if optimize_for == "Speed":                                     optim_target = "default"
    if optimize_for == "Maximized (Potentially Very slow startup)": optim_target = "max-autotune"
    set_config("optimize_for", optim_target)
    
    vampire_log.debug(f"Updated optimization config {compilation_method} {optim_target}")

def make_optimization_settings():
    with gr.Blocks() as interface:
        
        with gr.Box():
            gr.Label(value="""Use a torch compiler to dramatically increase generation speed. There's some caveats though, changing some parameters might trigger a re-compilation.
                     This is an experimental torch feature, so be warned that you might have issues when using this.""")
            with gr.Row():
                compilation_method = gr.Dropdown(choices=["None", "inductor", "eager", "aot_eager"], value="None", label="Use Compilation Method")
                optimize_for = gr.Dropdown(choices=["Memory Efficiency", "Speed", "Maximized (Potentially Very slow startup)"], value="Speed", label="Optimize For (Inductor only)")
        
            ui_items = [compilation_method, optimize_for]
    
    comp_dict = get_component_dictionary(locals())
    save, load, default = make_config_functions(OPTIMIZATION_TAB_NAME, comp_dict, update_optimiazation_config)
    
    interface.load(load, inputs=ui_items, outputs=ui_items)
    
    with gr.Row():
        save_button = gr.Button(value="Save Config")
        save_button.click(fn=save, inputs=ui_items, outputs=None)
        
        default_button = gr.Button(value="Return to default settings (Not saved until you hit save configs.)")
        default_button.click(fn=default, inputs=ui_items, outputs=ui_items)
    
    return interface