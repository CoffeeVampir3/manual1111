from shared.log import vampire_log

global RUNNING_CONFIG
RUNNING_CONFIG = {}

def get_config(config_name):
    return RUNNING_CONFIG.get(config_name)

def set_config(config_name, value):
    RUNNING_CONFIG[config_name] = value
    
def debug_config():
    for x,y in RUNNING_CONFIG.items():
        vampire_log.debug(f"{x} {y}")