from shared.log import vampire_log

RUNNING_CONFIG = {}

def get_config(config_name):
    global RUNNING_CONFIG
    return RUNNING_CONFIG.get(config_name)

def set_config(config_name, value):
    global RUNNING_CONFIG
    RUNNING_CONFIG[config_name] = value
    
def debug_config():
    global RUNNING_CONFIG
    for x,y in RUNNING_CONFIG.items():
        vampire_log.debug(f"{x} {y}")