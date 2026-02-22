import gradio as gr
from vidsearch import analyze_vid, store_data, ask_llama


def chat(message, history):
    reply = ask_llama(message)
    return reply

result = analyze_vid(r"video file here")
store_data("video_title", result) # replace vid title

gr.ChatInterface(chat, title = "VidSearch 🔍").launch(theme=gr.themes.Glass())
