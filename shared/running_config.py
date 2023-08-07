global RUNNING_CONFIG
RUNNING_CONFIG = {}

def get_config(config_name):
    return RUNNING_CONFIG.get(config_name)

def set_config(config_name, value):
    RUNNING_CONFIG[config_name] = value