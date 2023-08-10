from pathlib import Path
import gradio as gr
import os, json
 
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
            
    return results