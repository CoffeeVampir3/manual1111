from interactions import slash_command, SlashContext, slash_option, OptionType, SlashCommandChoice, Task, listen, IntervalTrigger
from interactions import Client, Intents, listen
from concurrent.futures import ThreadPoolExecutor
from mechanisms.t2i import run_t2i
from mechanisms.pipe_utils import unload_current_pipe
from shared.scheduler_utils import get_available_scheduler_names
from datetime import datetime
from PIL import Image
import asyncio, io, tempfile
import sys, os

with open('token.txt', 'r') as file:
    TOKEN = file.read().strip()
bot = Client(intents=Intents.DEFAULT, token=TOKEN)

MODEL_PATH = os.path.abspath(sys.argv[1])
global last_generation_time
last_generation_time = datetime.timestamp(datetime.now())

work_queue = asyncio.Queue()
async def worker():
    while True:
        global last_generation_time
        ctx, prompt, cfg, steps, width, height, scheduler = await work_queue.get()

        #safeguard against staling out mid run
        last_generation_time = datetime.timestamp(datetime.now())
        def run_t2i_generator():
            return list(run_t2i(
                MODEL_PATH,
                prompt, "",
                -1, cfg, steps, width, height,
                1, 1, scheduler
            ))

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            results = await loop.run_in_executor(pool, run_t2i_generator)

        image_data = results[0] if results else None
        image_data = image_data[0] if image_data else None

        if image_data:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file:
                image_data.save(temp_file.name, format='PNG') # Save the Image object to the temporary file
                await ctx.send(file=temp_file.name, filename="image.png")
                last_generation_time = datetime.timestamp(datetime.now())

        work_queue.task_done()
        
@Task.create(IntervalTrigger(minutes=1))
async def unload_if_unused_recently():
    global last_generation_time
    current_time = datetime.timestamp(datetime.now())
    print(f"Stale watchdog - Last generation was {current_time - last_generation_time:.2f} seconds ago.")
    if current_time - last_generation_time >= 120:
        unload_current_pipe()
        print("Unloaded any stale pipes!")

scheduler_choices = [SlashCommandChoice(name=key, value=key) for key in get_available_scheduler_names()]
@slash_command(name="t2i", description="John Rambonius")
@slash_option(
    name="prompt",
    description="The prompt.",
    required=True,
    opt_type=OptionType.STRING
)
@slash_option(
    name="cfg",
    description="Classifier free guidance value",
    required=False,
    opt_type=OptionType.NUMBER
)
@slash_option(
    name="steps",
    description="Steps",
    required=False,
    opt_type=OptionType.INTEGER,
    min_value=1,
    max_value=35
)
@slash_option(
    name="width",
    description="The width",
    required=False,
    opt_type=OptionType.INTEGER,
    min_value=512,
    max_value=2048
)
@slash_option(
    name="height",
    description="The height",
    required=False,
    opt_type=OptionType.INTEGER,
    min_value=512,
    max_value=2048
)
@slash_option(
    name="scheduler",
    description="Scheduler",
    required=False,
    opt_type=OptionType.STRING,
    choices=scheduler_choices
)
async def my_command_function(ctx: SlashContext, prompt, cfg=8.0, steps=20, width=1024, height=1024, scheduler="EulerDiscrete"):
    await ctx.defer() # Allows > 3 second duration for responses
    await work_queue.put((ctx, prompt, cfg, steps, width, height, scheduler))

@listen()
async def on_startup():
    unload_if_unused_recently.start()
    asyncio.create_task(worker())
    
bot.start()
