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
        
def load_json_impl(dest, **kwargs):
    with open(dest, 'r') as f:
        loaded_args = json.load(f)
    
    if not kwargs:
        return loaded_args
    
    results = []
    for key, default_value in kwargs.items():
        if key in loaded_args:
            results.append(gr.update(value=loaded_args[key]))
        else:
            results.append(gr.skip())
    
    if len(results) == 1:
        return results[0] #Not sure why this is needed, but it is.
    return results
        
def load_json_configs(tab_name, **kwargs):
    dir_path = Path("./configs")
    file_name = f"{tab_name}.json"
    dest = (dir_path/file_name)
    
    if not dest.exists():
        return load_default_json_configs(tab_name, **kwargs)

    return load_json_impl(dest, **kwargs)

def load_default_json_configs(tab_name, **kwargs):
    dir_path = Path("./configs/defaults")
    file_name = f"{tab_name}.json"
    dest = (dir_path/file_name)
    
    if not dest.exists():
        return [gr.skip()] * len(kwargs)
    
    return load_json_impl(dest, **kwargs)

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

def save_or_load_gradio_values(op, tab_name, local_components_dict, update_config_func, *resolved_component_list):
    ui_state = {}
    for key, value in zip(local_components_dict, resolved_component_list):
        ui_state[key] = value
        
    if op == "load":
        if update_config_func:
            #With no arguments, update func will load from file.
            update_config_func()
        return load_json_configs(tab_name, **ui_state)
    elif op == "default":
        return load_default_json_configs(tab_name, **ui_state)
    
    if update_config_func:
        #With arguments, update func will update from the gradio components
        update_config_func(**ui_state)
    save_json_configs(tab_name, **ui_state)
    
def make_config_functions(tab_name, local_components_dict, update_config_func):
    load = partial(save_or_load_gradio_values, "load", tab_name, local_components_dict, update_config_func)
    save = partial(save_or_load_gradio_values, "save", tab_name, local_components_dict, update_config_func)
    default = partial(save_or_load_gradio_values, "default", tab_name, local_components_dict, update_config_func)
    return save, load, default