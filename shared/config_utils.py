from pathlib import Path
import gradio as gr
import os, json
from gradio.components import Component as GradioComponent
from functools import partial
 
def save_json_configs(tab_name, **kwargs):
    if not os.path.exists('configs'):
        os.makedirs('configs', exist_ok=True)
        
    with open(f'configs/{tab_name}.json', 'w') as f:
        json.dump(kwargs, f)
        
def load_json_configs(tab_name, **kwargs):
    dir_path = Path("./configs")
    file_name = f"{tab_name}.json"
    dest = (dir_path/file_name)
    
    if not dest.exists():
        return [gr.skip()] * len(kwargs)

    with open(dest, 'r') as f:
        loaded_args = json.load(f)
    
    results = []
    for key, default_value in kwargs.items():
        if key in loaded_args:
            results.append(gr.update(value=loaded_args[key]))
        else:
            results.append(gr.skip())
    
    if len(results) == 1:
        return results[0] #Not sure why this is needed, but it is.
    return results

def get_dict_slice(comp_dict, key):
    return {key: comp_dict[key]} if key in comp_dict else None

def get_component_dictionary(comp_dict):
    final_items = {}
    for x,y in comp_dict.items():
        if isinstance(y, GradioComponent):
            val = get_dict_slice(comp_dict, x)
            if val:
                final_items.update(val)
    return final_items

def save_or_load_gradio_values(load, tab_name, local_components_dict, update_config_func, *resolved_component_list):
    ui_state = {}
    for key, value in zip(local_components_dict, resolved_component_list):
        ui_state[key] = value
        
    if update_config_func:
        update_config_func(**ui_state)
        
    if load:
        return load_json_configs(tab_name, **ui_state)

    save_json_configs(tab_name, **ui_state)
    
def get_config_save_load(tab_name, local_components_dict, update_config_func):
    load = partial(save_or_load_gradio_values, True, tab_name, local_components_dict, update_config_func)
    save = partial(save_or_load_gradio_values, False, tab_name, local_components_dict, update_config_func)
    return save, load