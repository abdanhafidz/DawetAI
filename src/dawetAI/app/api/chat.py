import gradio as gr
from inference import predict
gr.ChatInterface(
    fn=predict, 
    type="messages"
).launch()