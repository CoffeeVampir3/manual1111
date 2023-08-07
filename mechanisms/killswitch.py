class KillswitchEngaged(Exception):
    pass

global KILLSWITCH
KILLSWITCH = False

def killswitch_callback(step, t, latents):
    global KILLSWITCH
    if KILLSWITCH:
        KILLSWITCH = False
        raise KillswitchEngaged("")

def killswitch_engage():
    global KILLSWITCH
    KILLSWITCH = True