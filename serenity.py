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
    max_new_tokens=512,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

# SYSTEM_PROMPT = (
#     "You are Serenity, an AI wellness companion. When a user expresses sadness or feeling low, "
#     "acknowledge their feelings gently, validate their experience, and offer a caring, open-ended question or coping suggestion. "
#     "Do NOT give medical diagnoses.\n"
#     "Example:\n"
#     "User: I'm overwhelmed and sad.\n"
#     "Assistant: I'm sorry you're feeling this way. Would you like to talk about what triggered these feelings, or try a simple breathing exercise together?\n"
# )

SYSTEM_PROMPT = (
    "You are Serenity, an AI wellness companion and supportive conversation partner. "
    "Your role is to help users express and reflect on their emotions, especially when they are feeling sad, low, anxious, or stressed. "
    "Whenever a user shares a negative feeling, do the following in your reply:\n"
    "1. Gently acknowledge and validate their emotion by name (e.g., 'I'm sorry you're feeling anxious' or 'It's understandable to feel this way sometimes').\n"
    "2. Offer a *specific*, non-clinical suggestion or open-ended question that invites them to explore their feelings or self-care (e.g., 'Would you like to talk more about what happened?', 'Is there something that usually helps you feel a little better?', or 'Sometimes writing your feelings down can be helpful. Would you like to try a quick journaling exercise?').\n"
    "3. Vary your responses and do NOT repeat the same wording in every message.\n"
    "4. Avoid generic phrases like 'I'm here for you' or 'How are you feeling today?' unless the user is initiating the first message.\n"
    "5. Do NOT give any medical or diagnostic advice, and always encourage users to reach out to a human professional if they mention being in crisis.\n"
    "6. Keep your responses concise (1-3 sentences) and warm, with no lists or bullet points.\n"
    "7. If the user expresses positive feelings, celebrate them and ask a gentle follow-up to help them reflect on what contributed to that feeling.\n"
    "If the user asks for resources or urgent help, politely remind them that you are an AI companion and not a substitute for professional help, and suggest contacting a trusted person or local helpline."
)




def build_prompt(history, user_message):
    prompt = SYSTEM_PROMPT + "\n"
    for user, assistant in history:
        prompt += f"User: {user}\nAssistant: {assistant}\n"
    prompt += f"User: {user_message}\nAssistant:"
    return prompt

def generate_reply(user_message, history):
    prompt = build_prompt(history or [], user_message)
    output = generator(prompt, return_full_text=False)[0]["generated_text"]
    reply = output.split("User:")[0].strip()
    return reply

def respond(user_message, history):
    history = history or []
    assistant_reply = generate_reply(user_message, history)
    history.append((user_message, assistant_reply))
    return "", history

def reset_chat():
    return []

theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="indigo",
    neutral_hue="slate",
).set(
    body_background_fill="linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%)",
    block_background_fill="rgba(255,255,255,0.96)",
    block_shadow="0 4px 32px 0 rgba(136, 62, 245, 0.13)",
    block_border_width="0px",
    block_radius="28px",
    button_secondary_background_fill="#d1c4e9",
    button_secondary_text_color="#4527a0"
)


with gr.Blocks(theme=theme, title="Serenity – AI Wellness Assistant", css="""
    #serenity-header { 
        text-align: center; 
        font-size: 2.5rem; 
        font-weight: bold;
        background: linear-gradient(90deg, #bb86fc 30%, #8ec5fc 70%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -0.5em;
    }
    .gr-chatbot { 
        border-radius: 2rem !important; 
        box-shadow: 0 6px 36px 0 #b39ddb33;
        background: #fff7fb;
    }
    .gr-chat-message { 
        font-size: 1.08rem; 
    }
""") as demo:
    gr.Markdown('<div id="serenity-header">🧘‍♂️ Serenity</div>')
    gr.Markdown(
        """
        <div style="text-align:center; margin-bottom: 0.5em;">
            <span style="font-size:1.1em;">
            <strong>AI Wellness Companion (non‑clinical)</strong>
            </span><br>
            <span style="font-size:0.93em; color:#7454b4">
            Your private, gentle space to reflect.<br>
            <em>If you're in crisis, please seek professional help or call your local helpline.</em>
            </span>
        </div>
        """)
    chatbot = gr.Chatbot(
        avatar_images=("https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
                      "https://cdn-icons-png.flaticon.com/512/4185/4185633.png"),
        bubble_full_width=False,
        render_markdown=True,
        show_copy_button=True,
        height=440,
        label="Serenity Wellness Chat"
    )
    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="💬 How are you feeling today? (Press Enter to share)",
            container=False,
            scale=9,
            elem_id="serenity-input"
        )
        clear_btn = gr.Button("🧹 Clear", scale=1, elem_id="clear-btn")

    state = gr.State([])

    txt.submit(respond, inputs=[txt, state], outputs=[txt, chatbot])
    clear_btn.click(reset_chat, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
