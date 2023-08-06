from diffusers import (
    DDIMScheduler,
    DDIMParallelScheduler,
    DDPMScheduler,
    DDPMParallelScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    DPMSolverSinglestepScheduler,
    UniPCMultistepScheduler,
)

def get_available_schedulers():
    schedulers = {
        "HeunDiscrete": HeunDiscreteScheduler, #1/2
        "KDPM2Discrete": KDPM2DiscreteScheduler, #1/2
        "KDPM2-A-Discrete": KDPM2AncestralDiscreteScheduler, #3
        "EulerDiscrete": EulerDiscreteScheduler, #4
        "UniPCMulti": UniPCMultistepScheduler, #5
        "DDPMParallel": DDPMParallelScheduler, #6
        "Euler-A-Discrete": EulerAncestralDiscreteScheduler, #7
        "DDPM": DDPMScheduler, #Ranked 8
        "DDIMParallel": DDIMParallelScheduler, #Ranked 9
        "PNDM": PNDMScheduler, #Ranked 10
        "DPMSolverSingle": DPMSolverSinglestepScheduler, #Ranked 11
        "DDIM": DDIMScheduler, #Ranked 12
    }
    return schedulers
    
def get_available_scheduler_names():
    return get_available_schedulers().keys()

def get_scheduler_by_name(name):
    return get_available_schedulers()[name]