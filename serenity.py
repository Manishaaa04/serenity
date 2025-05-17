import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Use TinyLlama-1.1B-Chat model
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,  # -1 for CPU, 0 for GPU if available
    max_new_tokens=256,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

SYSTEM_PROMPT = (
    "You are Serenity, a supportive AI wellness assistant. "
    "Provide empathetic, non-clinical guidance and encourage reflection. "
    "Do NOT give medical diagnoses."
)

def build_prompt(history, user_message):
    prompt = SYSTEM_PROMPT + "\n"
    for user, assistant in history:
        prompt += f"User: {user}\nAssistant: {assistant}\n"
    prompt += f"User: {user_message}\nAssistant:"
    return prompt

# def generate_reply(user_message, history):
#     prompt = build_prompt(history or [], user_message)
#     out = generator(prompt, return_full_text=False)[0]["generated_text"]
#     return out.strip()
def generate_reply(user_message, history):
    prompt = build_prompt(history or [], user_message)
    output = generator(prompt, return_full_text=False)[0]["generated_text"]
    # Split output at 'User:' and return only the first part (the assistant reply)
    reply = output.split("User:")[0].strip()
    return reply


def respond(user_message, history):
    history = history or []
    assistant_reply = generate_reply(user_message, history)
    history.append((user_message, assistant_reply))
    return "", history

def reset_chat():
    return []

with gr.Blocks(title="Serenity – AI Wellness Assistant") as demo:
    gr.Markdown(
        """
        # Serenity
        **AI Wellness Companion (non‑clinical)**  
        If you're in crisis, please seek professional help or call your local helpline.
        """
    )
    chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="How are you feeling today?",
            container=False,
            scale=9,
        )
        clear_btn = gr.Button("Clear", scale=1)

    state = gr.State([])

    txt.submit(respond, inputs=[txt, state], outputs=[txt, chatbot])
    clear_btn.click(reset_chat, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
