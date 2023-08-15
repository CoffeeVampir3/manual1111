from interactions import slash_command, SlashContext, slash_option, OptionType, SlashCommandChoice, Task, listen
from interactions import Client, Intents, listen
from concurrent.futures import ThreadPoolExecutor
from mechanisms.t2i import run_t2i
from shared.scheduler_utils import get_available_scheduler_names
from PIL import Image
import asyncio, io, tempfile
import sys, os

with open('token.txt', 'r') as file:
    TOKEN = file.read().strip()
bot = Client(intents=Intents.DEFAULT, token=TOKEN)

MODEL_PATH = os.path.abspath(sys.argv[1])
print(MODEL_PATH)

work_queue = asyncio.Queue()
async def worker():
    while True:
        ctx, prompt, cfg, steps, width, height, scheduler = await work_queue.get()

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

        work_queue.task_done()

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
    asyncio.create_task(worker())
    
bot.start()
