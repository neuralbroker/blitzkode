#!/usr/bin/env python3
"""
Professional Web Interface for BlitzKode
With streaming, syntax highlighting, and modern UI
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig
import gradio as gr
import re
import html
from threading import Thread

MODEL_PATH = "C:/Dev/Projects/BlitzKode/checkpoints/dpo-v1/final"

CSS = """
:root {
    --primary: #00d4ff;
    --secondary: #7c3aed;
    --bg-dark: #0a0a0f;
    --bg-card: #12121a;
    --text-main: #e4e4e7;
    --text-muted: #71717a;
    --border: #27272a;
    --success: #22c55e;
    --error: #ef4444;
}

body {
    background: var(--bg-dark) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}

.main-header {
    text-align: center;
    padding: 20px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 30px;
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.main-header p {
    color: var(--text-muted);
    margin-top: 10px;
}

.chat-container {
    background: var(--bg-card);
    border-radius: 16px;
    border: 1px solid var(--border);
    overflow: hidden;
}

.message-row {
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
}

.message-row.user {
    background: rgba(0, 212, 255, 0.05);
}

.message-row.assistant {
    background: rgba(124, 58, 237, 0.05);
}

.message-label {
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}

.message-row.user .message-label {
    color: var(--primary);
}

.message-row.assistant .message-label {
    color: var(--secondary);
}

.message-content {
    line-height: 1.7;
    font-size: 0.95rem;
}

.code-block {
    background: #0d0d12;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
    overflow-x: auto;
    border: 1px solid var(--border);
}

.code-block pre {
    margin: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.85rem;
    line-height: 1.5;
}

.code-block code {
    color: #a5b4fc;
}

.input-area {
    background: var(--bg-card);
    border-top: 1px solid var(--border);
    padding: 20px;
}

.input-area textarea {
    background: var(--bg-dark) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-main) !important;
    font-size: 1rem !important;
    padding: 16px !important;
}

.input-area textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1) !important;
}

.send-btn {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    color: white !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

.send-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 212, 255, 0.3) !important;
}

.settings-panel {
    background: var(--bg-card);
    border-radius: 12px;
    border: 1px solid var(--border);
    padding: 20px;
    margin-bottom: 20px;
}

.settings-panel h3 {
    color: var(--text-main);
    margin-bottom: 16px;
    font-size: 1rem;
}

.slider-container {
    margin: 12px 0;
}

.slider-container label {
    color: var(--text-muted);
    font-size: 0.85rem;
}

.status-bar {
    background: var(--bg-card);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.85rem;
    color: var(--text-muted);
}

.status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 2px solid var(--border);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.examples-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 12px;
    margin-top: 16px;
}

.example-btn {
    background: var(--bg-dark) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    text-align: left !important;
    transition: all 0.2s !important;
}

.example-btn:hover {
    border-color: var(--primary) !important;
    color: var(--text-main) !important;
}
"""

def highlight_code(code):
    code = html.escape(code)
    keywords = r'\b(def|class|if|elif|else|for|while|return|import|from|as|try|except|finally|with|lambda|and|or|not|in|is|True|False|None|print|print|range|len|list|dict|set|tuple|int|str|float|bool)\b'
    code = re.sub(keywords, r'<span class="keyword">\1</span>', code)
    code = code.replace('\n', '<br>')
    code = code.replace('    ', '&nbsp;&nbsp;&nbsp;&nbsp;')
    return f'<pre><code>{code}</code></pre>'

def format_response(text):
    parts = []
    current_pos = 0
    
    code_pattern = r'```(\w+)?\n(.*?)```'
    for match in re.finditer(code_pattern, text, re.DOTALL):
        if match.start() > current_pos:
            parts.append(f'<div class="text-block">{html.escape(text[current_pos:match.start()])}</div>')
        
        lang = match.group(1) or 'python'
        code = match.group(2).strip()
        parts.append(f'<div class="code-block"><div class="lang-label">{lang}</div><pre><code>{html.escape(code)}</code></pre></div>')
        current_pos = match.end()
    
    if current_pos < len(text):
        parts.append(f'<div class="text-block">{html.escape(text[current_pos:])}</div>')
    
    if not parts:
        parts.append(f'<div class="text-block">{html.escape(text)}</div>')
    
    return '\n'.join(parts)

print("Loading BlitzKode Model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

peft_config = PeftConfig.from_pretrained(MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

generation_config = GenerationConfig(
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

print("BlitzKode Ready!")

def generate(prompt, temperature, max_tokens, history=None):
    full_prompt = f"<|im_start|>system\nYou are BlitzKode, an expert coding assistant. Provide clear, efficient, and well-documented code solutions.<|im_end|>\n"
    
    if history:
        for user_msg, assistant_msg in history:
            full_prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"
    
    full_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            temperature=temperature,
            max_new_tokens=int(max_tokens),
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(full_prompt):].strip()
    
    return response

EXAMPLES = [
    "Write a Python function to implement binary search",
    "Create a REST API endpoint using Flask",
    "Implement a linked list data structure in Python",
    "Write a Python decorator for timing function execution",
    "Implement quicksort algorithm in Python",
    "Create a Python class for a bank account",
]

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div class="main-header">
        <h1>⚡ BlitzKode</h1>
        <p>Your AI-Powered Coding Assistant</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                chatbot = gr.Chatbot(
                    height=500,
                    type="messages",
                )
                
                with gr.Row():
                    prompt_input = gr.Textbox(
                        label="",
                        placeholder="Ask me to write code, explain algorithms, or help with programming questions...",
                        lines=3,
                        container=False,
                    )
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", size="lg")
        
        with gr.Column(scale=1):
            with gr.Group():
                gr.HTML("<h3>⚙️ Settings</h3>")
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    label="Temperature (Creativity)",
                )
                
                max_tokens = gr.Slider(
                    minimum=128,
                    maximum=2048,
                    value=1024,
                    step=128,
                    label="Max Tokens",
                )
                 
                gr.HTML("<h3>💡 Examples</h3>")
                for i, ex in enumerate(EXAMPLES):
                    btn = gr.Button(ex, variant="secondary", size="sm")
                    btn.click(
                        fn=lambda e=ex: e,
                        inputs=[],
                        outputs=prompt_input,
                    )
    
    gr.HTML("""
    <div class="status-bar">
        <div class="status-indicator"></div>
        <span>Model Ready • RTX 4060 Laptop GPU</span>
    </div>
    """)
    
    def respond(prompt, history, temperature, max_tokens):
        if not prompt.strip():
            return history, ""
        
        response = generate(prompt, temperature, max_tokens, history)
        history.append((prompt, response))
        return history, ""
    
    submit_btn.click(
        respond,
        inputs=[prompt_input, chatbot, temperature, max_tokens],
        outputs=[chatbot, prompt_input],
    )
    
    prompt_input.submit(
        respond,
        inputs=[prompt_input, chatbot, temperature, max_tokens],
        outputs=[chatbot, prompt_input],
    )
    
    clear_btn.click(lambda: [], outputs=chatbot)

demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
