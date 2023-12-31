from diffusers import (
    DDIMScheduler,
    DDIMParallelScheduler,
    DDPMScheduler,
    DDPMParallelScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)

def get_available_schedulers():
    schedulers = {
        "DDIM": DDIMScheduler,
        "DDIMParallel": DDIMParallelScheduler,
        "DDPM": DDPMScheduler,
        "DDPMParallel": DDPMParallelScheduler,
        "DEISMultistep": DEISMultistepScheduler,
        "DPMSolverMultistep": DPMSolverMultistepScheduler,
        "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
        "EulerAncestralDiscrete": EulerAncestralDiscreteScheduler,
        "EulerDiscrete": EulerDiscreteScheduler,
        "HeunDiscrete": HeunDiscreteScheduler,
        "KDPM2AncestralDiscrete": KDPM2AncestralDiscreteScheduler,
        "KDPM2Discrete": KDPM2DiscreteScheduler,
        "PNDM": PNDMScheduler,
        "UniPCMultistep": UniPCMultistepScheduler,
    }
    return schedulers
    
def get_available_scheduler_names():
    return get_available_schedulers().keys()

def get_scheduler_by_name(name):
    return get_available_schedulers()[name]

def get_name_by_scheduler(scheduler):
    schedulers = get_available_schedulers()
    
    for name, scheduler_cls in schedulers.items():
        if scheduler_cls == scheduler:
            return str(name)
    return None
