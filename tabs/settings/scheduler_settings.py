import gradio as gr
import os
from shared.running_config import set_config
from functools import partial
from shared.config_utils import get_config_save_load, get_component_dictionary

def update_config(scheduler_name, **kwargs):
    scheduler_settings = {
        "scheduler_name": scheduler_name,
        "trained_betas": False,
        "timestep_spacing": False,
    }
    
    scheduler_settings.update(kwargs)
    set_config(scheduler_name, scheduler_settings)
    print(f"Updated {scheduler_name} configs!")

def make_singular_scheduler_tab(scheduler_name, base_scheduler_settings, interface):
    with gr.Tab(label=scheduler_name) as scheduler_tab:
        with gr.Row():
            use_karras_sigmas = gr.Checkbox(label="Use Karras Sigmas", value=base_scheduler_settings["use_karras_sigmas"])
            clip_sample = gr.Checkbox(label="Clip Sample", value=base_scheduler_settings["clip_sample"])
            set_alpha_to_one = gr.Checkbox(label="Set Alpha to One", value=base_scheduler_settings["set_alpha_to_one"])
            skip_prk_steps = gr.Checkbox(label="Skip PRK Steps", value=base_scheduler_settings["skip_prk_steps"])

        with gr.Row():
            beta_schedule = gr.Dropdown(choices=["linear", "scaled_linear"], value=base_scheduler_settings["beta_schedule"], label="Beta Schedule")
            beta_start = gr.Number(value=base_scheduler_settings["beta_start"], label="Beta Start")
            beta_end = gr.Number(value=base_scheduler_settings["beta_end"], label="Beta End")

        with gr.Row():
            interpolation_type = gr.Dropdown(choices=["linear"], value=base_scheduler_settings["interpolation_type"], label="Interpolation Type")
            prediction_type = gr.Dropdown(choices=["epsilon"], value=base_scheduler_settings["prediction_type"], label="Prediction Type")
            sample_max_value = gr.Number(value=base_scheduler_settings["sample_max_value"], label="Sample Max Value")
            steps_offset = gr.Number(value=base_scheduler_settings["steps_offset"], label="Steps Offset", precision=0)
    
    ui_items = [
        use_karras_sigmas,
        clip_sample,
        set_alpha_to_one,
        skip_prk_steps,
        beta_schedule,
        beta_start,
        beta_end,
        interpolation_type,
        prediction_type,
        sample_max_value,
        steps_offset
    ]
    
    scheduler_configurator = partial(update_config, scheduler_name)
    comp_dict = get_component_dictionary(locals())
    save, load = get_config_save_load(scheduler_name, comp_dict, scheduler_configurator)
    
    interface.load(load, inputs=ui_items, outputs=ui_items)
    
    save_button = gr.Button(value="Save Config")
    save_button.click(fn=save, inputs=ui_items, outputs=None)
    
    return scheduler_tab

default_scheduler_values = {
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "interpolation_type": "linear",
    "prediction_type": "epsilon",
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "trained_betas": None,
    "use_karras_sigmas": False
}

def make_scheduler_settings():
    with gr.Blocks() as interface:
        make_singular_scheduler_tab("EulerDiscrete", default_scheduler_values, interface)
    
    return interface