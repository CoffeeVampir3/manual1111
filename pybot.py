from interactions import slash_command, SlashContext, slash_option, OptionType, SlashCommandChoice, Task, listen, IntervalTrigger, Attachment
from interactions import Client, Intents, listen
from concurrent.futures import ThreadPoolExecutor
from mechanisms.run_pipe import run_t2i, run_i2i
from mechanisms.pipe_utils import unload_current_pipe
from shared.scheduler_utils import get_available_scheduler_names
from datetime import datetime
from PIL import Image
import aiohttp
import asyncio, io, tempfile
import sys, os

with open('token.txt', 'r') as file:
    TOKEN = file.read().strip()
bot = Client(intents=Intents.DEFAULT, token=TOKEN)

MODEL_PATH = os.path.abspath(sys.argv[1])
global last_generation_time
last_generation_time = datetime.timestamp(datetime.now())

global relevant_posts
relevant_posts = set()

work_queue = asyncio.Queue()
async def worker():
    while True:
        global last_generation_time
        global relevant_posts
        ctx, prompt, cfg, steps, width, height, scheduler, image, intensity = await work_queue.get()

        #safeguard against staling out mid run
        last_generation_time = datetime.timestamp(datetime.now())
        if image is None:
            def run_generator():
                return list(run_t2i(
                    MODEL_PATH, width, height,
                    prompt, "",
                    -1, cfg, steps,
                    1, 1, scheduler
                ))
        else:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(image.url) as response:
                        initialization_data = Image.open(io.BytesIO(await response.read())).convert("RGB")
                        def run_generator():
                            return list(run_i2i(
                                MODEL_PATH, initialization_data, intensity,
                                prompt, "",
                                -1, cfg, steps,
                                1, 1, scheduler
                            ))
            except Exception as e:
                print(f"Fucked up. {e}")
                await ctx.send(content="Generation failed Q_Q")
                work_queue.task_done()
                return

        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                results = await loop.run_in_executor(pool, run_generator)
        except Exception as e:
            print(f"Fucked up. {e}")
            await ctx.send(content="Generation failed Q_Q")
            work_queue.task_done()
            return

        image_data = results[0] if results else None
        image_data = image_data[0] if image_data else None

        if image_data:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file:
                image_data.save(temp_file.name, format='PNG') # Save the Image object to the temporary file
                message = await ctx.send(file=temp_file.name, filename="image.png")
                relevant_posts.add(message.id)
                print(f"Added: {message.id}")
                await message.add_reaction(":heart:")
                await message.add_reaction(":fire:")
                await message.add_reaction(":skull:")
                await message.add_reaction(":peach:")
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
@slash_command(name="generate", description="John Rambonius")
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
    name="intensity",
    description="The intensity",
    required=False,
    opt_type=OptionType.NUMBER,
    min_value=0.01,
    max_value=1.0
)
@slash_option(
    name="scheduler",
    description="Scheduler",
    required=False,
    opt_type=OptionType.STRING,
    choices=scheduler_choices
)
@slash_option(
    name="image",
    description="image",
    required=False,
    opt_type=OptionType.ATTACHMENT,
)
async def my_command_function(ctx: SlashContext, prompt, cfg=8.0, steps=20, width=1024, height=1024, scheduler="EulerDiscrete", image:Attachment=None, intensity=0.7):
    await ctx.defer() # Allows > 3 second duration for responses
    await work_queue.put((ctx, prompt, cfg, steps, width, height, scheduler, image, intensity))

@listen()
async def on_startup():
    unload_if_unused_recently.start()
    asyncio.create_task(worker())

from interactions.api.events import MessageReactionAdd
@listen(MessageReactionAdd)
async def listen_to_reactions(event: MessageReactionAdd):
    global relevant_posts
    is_in = event.message.id in relevant_posts
    
    if not is_in:
        return
    
    #Debugging
    # print(f"Author: {event.author}")
    # print(f"Author id: {event.author.id}")
    # print(f"Channel: {event.reaction.channel}")
    # print(f"Message: {event.reaction.message}")
    # print(f"Message id: {event.message.id}")
    # print(f"In set? {is_in}")
    # print(f"Emoji: {event.reaction.emoji}")
    # print(f"Count: {event.reaction.count}")
    
bot.start()
