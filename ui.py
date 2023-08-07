import gradio as gr
import argparse
from tabs.t2i_tab import make_text_to_image_tab
from functools import partial

custom_css = (""".gradio-container {
    min-width: 100%;
    }""")

with gr.Blocks(css=custom_css) as interface:
    with gr.Tab("Text -> Image"):
        make_text_to_image_tab()
        
def get_cli_args():
    parser = argparse.ArgumentParser(description='SDXL Diffusion Webui.')
    
    parser.add_argument('-s', "--share", action='store_true', help='Make a sharable address over the internet via gradio proxy')
    parser.add_argument('-b', '--bind', action='store_true', help='Launch as a binding interface')
    
    args = parser.parse_args()
    return args

def bind_launch_args(interface, args):
    launch_func = partial(interface.launch, quiet=False)
    launch_func = partial(launch_func, share=args.share)
    if args.bind:
        launch_func = partial(launch_func, server_name="0.0.0.0")
        
    return launch_func

if __name__ == '__main__':
    args = get_cli_args()
    bound_launch = bind_launch_args(interface, args)
    interface.queue()
    bound_launch()