from omegaconf import OmegaConf
from pathlib import Path

def save_ui_config(tab_name, **kwargs):
    config = OmegaConf.create(kwargs)
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
    return list(config.values())