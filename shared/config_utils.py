from omegaconf import OmegaConf
from pathlib import Path
import gradio as gr

def save_ui_config(tab_name, **kwargs):
    config = OmegaConf.create(kwargs)
    
    num_arguments = len(kwargs)
    config.number_of_arguments = num_arguments
    dir_path = Path("./configs")
    file_name = f"{tab_name}.yaml"
    if not dir_path.exists():
        dir_path.mkdir()
    dest = (dir_path/file_name)
    OmegaConf.save(config=config, f=dest)
    return config

def load_ui_config(tab_name, blank_items):
    dir_path = Path("./configs")
    file_name = f"{tab_name}.yaml"
    dest = (dir_path/file_name)

    if not dest.exists():
        return {item: gr.skip() for item in blank_items} #Hack but it work

    config = OmegaConf.load(dest)
    
    num_args = config.get("number_of_arguments")
    if num_args:
        del config.number_of_arguments
        
    values = list(OmegaConf.to_container(config, resolve=True).values())
    num_values = len(values)
    if not num_args or num_args != num_values:
        print(f"Found an outdated/mismatched config for: {tab_name} - unable to load it.")
        return {item: gr.skip() for item in blank_items}
    
    return values