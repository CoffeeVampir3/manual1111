import gradio as gr
from tabs.t2i_tab import make_text_to_image_tab

custom_css = (""".gradio-container {
    min-width: 100%;
    }""")

with gr.Blocks(css=custom_css) as interface:
    with gr.Tab("Text -> Image"):
        make_text_to_image_tab()

interface.queue().launch(quiet=False)