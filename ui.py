import gradio as gr

# IMPORTANT: 
# If you saved the previous code in a file named 'main.py', keep this import.
# If you just ran the previous code in a cell above, comment out or remove this line!
from main import generate_interview_response

iface = gr.Interface(
    fn=generate_interview_response,
    inputs=gr.Textbox(
        label="Enter your Query", 
        placeholder="e.g., Infosys Python Developer interview questions", 
        lines=2
    ),
    outputs=gr.Textbox(
        label="Generated Interview Questions", 
        lines=30, 
        show_copy_button=True
    ),
    title="AI Placement Interview Assistant",
    description="Generates interview questions and short answers using RAG."
)

# In Google Colab, you MUST use share=True to generate a public URL 
# that you can click on to access the Gradio interface.
iface.launch(share=True)
