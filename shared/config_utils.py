from omegaconf import OmegaConf
from pathlib import Path

def save_ui_config(tab_name, **kwargs):
    config = OmegaConf.create(kwargs)
    
    num_arguments = len(kwargs)
    config.number_of_arguments = num_arguments+1
    dir_path = Path("./configs")
    file_name = f"{tab_name}_last_run.yaml"
    if not dir_path.exists():
        dir_path.mkdir()
    dest = (dir_path/file_name)
    OmegaConf.save(config=config, f=dest)
    return config

def load_ui_config(tab_name, model_path):
    dir_path = Path("./configs")
    file_name = f"{tab_name}_last_run.yaml"
    dest = (dir_path/file_name)

    if not dest.exists():
        return {model_path: ""} #Hack but it work

    config = OmegaConf.load(dest)
    
    values = list(config.values())
    num_values = len(values)
    if not config.get("number_of_arguments") or config.get("number_of_arguments") != num_values:
        return {model_path: ""}
    
    return values