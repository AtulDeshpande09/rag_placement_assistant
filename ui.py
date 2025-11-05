import gradio as gr
from main import generate_interview_response

iface = gr.Interface(
    fn=generate_interview_response,
    inputs=gr.Textbox(label="Enter your Query", placeholder="e.g., Infosys Python Developer interview questions", lines=2),
    outputs=gr.Textbox(label="Generated Interview Questions", lines=30, show_copy_button=True),
    title="AI Placement Interview Assistant",
    description="Generates interview questions and short answers using RAG."
)

iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
