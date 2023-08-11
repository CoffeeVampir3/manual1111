class KillswitchEngaged(Exception):
    pass

global KILLSWITCH
KILLSWITCH = False

def killswitch_callback(step, t, latents):
    global KILLSWITCH
    if KILLSWITCH:
        killswitch_reset()
        raise KillswitchEngaged("")

def killswitch_engage():
    global KILLSWITCH
    KILLSWITCH = True
    
def killswitch_reset():
    global KILLSWITCH
    KILLSWITCH = False