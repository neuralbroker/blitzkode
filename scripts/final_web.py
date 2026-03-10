#!/usr/bin/env python3
"""
BlitzKode - Final Production Web Interface
Professional UI with dark theme, streaming, and optimized performance
"""

import os
import sys
from pathlib import Path
import llama_cpp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Configuration
BLITZKODE_DIR = Path("C:/Dev/Projects/BlitzKode")
MODEL_PATH = BLITZKODE_DIR / "blitzkode.gguf"

if not MODEL_PATH.exists():
    print(f"ERROR: Model not found: {MODEL_PATH}")
    print("Please export model first: python scripts/export_gguf.py")
    sys.exit(1)

print("=" * 60)
print("BLITZKODE - OPTIMIZED INFERENCE SERVER")
print("=" * 60)
print(f"Model: {MODEL_PATH}")
print(f"Size: {MODEL_PATH.stat().st_size / 1e9:.2f} GB")

# Load model with maximum optimization
print("\n[LOADING MODEL WITH OPTIMIZATIONS]")

llm = llama_cpp.Llama(
    model_path=str(MODEL_PATH),
    n_gpu_layers=1,        # Use GPU
    n_ctx=2048,             # Context window
    n_threads=8,             # CPU threads
    n_batch=1024,           # Larger batch for speed
    verbose=False,
    use_mmap=True,
    use_mlock=False,
    rope_freq_base=10000.0,  # RoPE frequency
    rope_freq_scale=1.0,
)

print("[MODEL READY]\n")

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optimized system prompt
SYSTEM_PROMPT = """<|im_start|>system
You are BlitzKode, an expert coding assistant. Provide clean, efficient, and well-documented code. Keep responses concise and focused. Only output code when explicitly asked.<|im_end|>"""

@app.get("/")
async def root():
    return HTMLResponse(get_html())

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    temperature = data.get("temperature", 0.3)  # Lower for more focused code
    max_tokens = data.get("max_tokens", 512)
    
    full_prompt = f"{SYSTEM_PROMPT}\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Optimized generation
    result = llm(
        full_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        stop=["<|im_end|>", "<|im_start|>"],
    )
    
    response = result["choices"][0]["text"].strip()
    return JSONResponse({"response": response})

@app.post("/generate_stream")
async def generate_stream(request: Request):
    """Streaming endpoint for real-time responses"""
    data = await request.json()
    prompt = data.get("prompt", "")
    temperature = data.get("temperature", 0.3)
    max_tokens = data.get("max_tokens", 512)
    
    full_prompt = f"{SYSTEM_PROMPT}\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    async def token_generator():
        try:
            for token in llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stream=True,
                stop=["<|im_end|>", "<|im_start|>"],
            ):
                if "choices" in token and len(token["choices"]) > 0:
                    text = token["choices"][0].get("text", "")
                    if text:
                        yield f"data: {text}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR]{str(e)}\n\n"
    
    from starlette.datastructures import URL
    from starlette.responses import StreamingResponse
    return StreamingResponse(token_generator(), media_type="text/event-stream")

def get_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BlitzKode - AI Coding Assistant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a24;
            --bg-hover: #22222e;
            --accent-primary: #00d4ff;
            --accent-secondary: #7c3aed;
            --accent-gradient: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
            --text-primary: #f4f4f5;
            --text-secondary: #a1a1aa;
            --text-muted: #71717a;
            --border: #27272a;
            --success: #22c55e;
            --error: #ef4444;
            --code-bg: #0d0d14;
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 16px;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        /* Header */
        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 0.875rem 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            backdrop-filter: blur(10px);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .logo-container {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .logo {
            width: 36px;
            height: 36px;
            background: var(--accent-gradient);
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.25rem;
            color: white;
        }
        
        .brand {
            font-size: 1.25rem;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.025em;
        }
        
        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.375rem 0.75rem;
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.2);
            border-radius: 999px;
            font-size: 0.8125rem;
            color: var(--success);
        }
        
        .status-dot {
            width: 6px;
            height: 6px;
            background: var(--success);
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(0.9); }
        }
        
        /* Main Layout */
        .main-container {
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        
        /* Sidebar */
        .sidebar {
            width: 280px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .sidebar-section {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .sidebar-title {
            font-size: 0.6875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            padding: 0 0.5rem;
        }
        
        .prompt-btn {
            display: flex;
            align-items: center;
            gap: 0.625rem;
            padding: 0.625rem 0.75rem;
            background: var(--bg-tertiary);
            border: 1px solid transparent;
            border-radius: var(--radius-md);
            color: var(--text-secondary);
            font-size: 0.8125rem;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
        }
        
        .prompt-btn:hover {
            background: var(--bg-hover);
            border-color: var(--border);
            color: var(--text-primary);
        }
        
        .prompt-icon {
            font-size: 1rem;
            opacity: 0.7;
        }
        
        /* Chat Area */
        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .message {
            max-width: 85%;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            align-self: flex-end;
        }
        
        .message.assistant {
            align-self: flex-start;
        }
        
        .message-bubble {
            padding: 1rem 1.25rem;
            border-radius: var(--radius-lg);
            line-height: 1.6;
        }
        
        .message.user .message-bubble {
            background: var(--accent-gradient);
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .message.assistant .message-bubble {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-bottom-left-radius: 4px;
        }
        
        .message-label {
            font-size: 0.6875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
            color: var(--text-muted);
        }
        
        .message.assistant .message-label {
            color: var(--accent-primary);
        }
        
        /* Code Blocks */
        pre {
            background: var(--code-bg);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 1rem;
            margin: 0.75rem 0;
            overflow-x: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8125rem;
            line-height: 1.5;
        }
        
        code {
            font-family: inherit;
            color: #a5b4fc;
        }
        
        p code {
            background: var(--bg-tertiary);
            padding: 0.125rem 0.375rem;
            border-radius: 4px;
            font-size: 0.875em;
        }
        
        /* Input Area */
        .input-container {
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
            padding: 1rem 1.5rem;
        }
        
        .input-wrapper {
            max-width: 900px;
            margin: 0 auto;
            display: flex;
            gap: 0.75rem;
            align-items: flex-end;
        }
        
        .input-box {
            flex: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 0.875rem 1rem;
            color: var(--text-primary);
            font-size: 0.9375rem;
            font-family: inherit;
            resize: none;
            min-height: 52px;
            max-height: 200px;
            line-height: 1.5;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        
        .input-box:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
        }
        
        .input-box::placeholder {
            color: var(--text-muted);
        }
        
        .send-button {
            background: var(--accent-gradient);
            border: none;
            border-radius: var(--radius-md);
            padding: 0.875rem 1.5rem;
            color: white;
            font-weight: 600;
            font-size: 0.9375rem;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
        }
        
        .send-button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3);
        }
        
        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        /* Settings Panel */
        .settings-panel {
            display: flex;
            gap: 1.5rem;
            padding: 0.75rem 1.5rem;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border);
            flex-wrap: wrap;
        }
        
        .setting-group {
            display: flex;
            align-items: center;
            gap: 0.625rem;
        }
        
        .setting-label {
            font-size: 0.8125rem;
            color: var(--text-secondary);
        }
        
        .setting-slider {
            width: 100px;
            height: 4px;
            accent-color: var(--accent-primary);
            cursor: pointer;
        }
        
        .setting-value {
            font-size: 0.8125rem;
            color: var(--text-primary);
            min-width: 35px;
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Loading State */
        .typing-indicator {
            display: inline-flex;
            gap: 4px;
            padding: 0.5rem;
        }
        
        .typing-indicator span {
            width: 6px;
            height: 6px;
            background: var(--accent-primary);
            border-radius: 50%;
            animation: bounce 1.4s ease-in-out infinite;
        }
        
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-6px); }
        }
        
        /* Welcome Message */
        .welcome-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 2rem;
        }
        
        .welcome-logo {
            width: 80px;
            height: 80px;
            background: var(--accent-gradient);
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            font-weight: 700;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 20px 60px rgba(0, 212, 255, 0.2);
        }
        
        .welcome-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .welcome-subtitle {
            color: var(--text-secondary);
            font-size: 1.0625rem;
            max-width: 400px;
            line-height: 1.6;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="logo-container">
            <div class="logo">B</div>
            <span class="brand">BlitzKode</span>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span>
            <span>Online</span>
        </div>
    </header>
    
    <div class="main-container">
        <aside class="sidebar">
            <div class="sidebar-section">
                <span class="sidebar-title">Quick Prompts</span>
                <button class="prompt-btn" onclick="setPrompt('Write binary search in Python')">
                    <span class="prompt-icon">&#128269;</span>
                    Binary Search
                </button>
                <button class="prompt-btn" onclick="setPrompt('Implement a Stack class')">
                    <span class="prompt-icon">&#128230;</span>
                    Stack Implementation
                </button>
                <button class="prompt-btn" onclick="setPrompt('Create a REST API with Flask')">
                    <span class="prompt-icon">&#127760;</span>
                    REST API
                </button>
                <button class="prompt-btn" onclick="setPrompt('Python decorator for timing')">
                    <span class="prompt-icon">&#9201;</span>
                    Timer Decorator
                </button>
                <button class="prompt-btn" onclick="setPrompt('Explain quicksort algorithm')">
                    <span class="prompt-icon">&#128161;</span>
                    Quicksort
                </button>
            </div>
        </aside>
        
        <main class="chat-area">
            <div class="settings-panel">
                <div class="setting-group">
                    <span class="setting-label">Temperature</span>
                    <input type="range" class="setting-slider" id="temp" min="0.1" max="1.0" step="0.1" value="0.3">
                    <span class="setting-value" id="temp-val">0.3</span>
                </div>
                <div class="setting-group">
                    <span class="setting-label">Max Tokens</span>
                    <input type="range" class="setting-slider" id="tokens" min="64" max="1024" step="64" value="512">
                    <span class="setting-value" id="tokens-val">512</span>
                </div>
            </div>
            
            <div class="messages-container" id="messages">
                <div class="welcome-container">
                    <div class="welcome-logo">B</div>
                    <h1 class="welcome-title">BlitzKode</h1>
                    <p class="welcome-subtitle">Your AI-powered coding assistant. Write code, debug, and learn programming concepts.</p>
                </div>
            </div>
            
            <div class="input-container">
                <div class="input-wrapper">
                    <textarea 
                        class="input-box" 
                        id="prompt" 
                        placeholder="Ask me to write code, explain algorithms, or help with programming..."
                        rows="1"
                    ></textarea>
                    <button class="send-button" id="send-btn" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </main>
    </div>

    <script>
        const messagesDiv = document.getElementById('messages');
        const promptInput = document.getElementById('prompt');
        const sendBtn = document.getElementById('send-btn');
        const tempSlider = document.getElementById('temp');
        const tokensSlider = document.getElementById('tokens');
        
        // Update slider displays
        tempSlider.oninput = () => document.getElementById('temp-val').textContent = tempSlider.value;
        tokensSlider.oninput = () => document.getElementById('tokens-val').textContent = tokensSlider.value;
        
        // Auto-resize textarea
        promptInput.oninput = () => {
            promptInput.style.height = 'auto';
            promptInput.style.height = Math.min(promptInput.scrollHeight, 200) + 'px';
        };
        
        // Send on Enter (without Shift)
        promptInput.onkeydown = (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        };
        
        function setPrompt(text) {
            promptInput.value = text;
            promptInput.style.height = 'auto';
            promptInput.style.height = promptInput.scrollHeight + 'px';
            promptInput.focus();
            hideWelcome();
        }
        
        function hideWelcome() {
            const welcome = document.querySelector('.welcome-container');
            if (welcome) welcome.remove();
        }
        
        function addMessage(role, content) {
            hideWelcome();
            
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = `
                <div class="message-bubble">
                    <div class="message-label">${role === 'user' ? 'You' : 'BlitzKode'}</div>
                    ${formatContent(content)}
                </div>
            `;
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function formatContent(text) {
            // Escape HTML
            text = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            
            // Format code blocks
            text = text.replace(/```(\\w*)\\n?([\\s\\S]*?)```/g, (match, lang, code) => {
                return `<pre><code>${code.trim()}</code></pre>`;
            });
            
            // Format inline code
            text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
            
            // Convert newlines to <br>
            text = text.replace(/\\n/g, '<br>');
            
            return `<div style="line-height: 1.7;">${text}</div>`;
        }
        
        async function sendMessage() {
            const prompt = promptInput.value.trim();
            if (!prompt) return;
            
            // Add user message
            addMessage('user', prompt);
            promptInput.value = '';
            promptInput.style.height = 'auto';
            
            // Show loading
            sendBtn.disabled = true;
            sendBtn.innerHTML = '<span class="typing-indicator"><span></span><span></span><span></span></span>';
            
            // Create assistant message placeholder
            hideWelcome();
            const assistantDiv = document.createElement('div');
            assistantDiv.className = 'message assistant';
            assistantDiv.innerHTML = `
                <div class="message-bubble">
                    <div class="message-label">BlitzKode</div>
                    <div id="response-content"><span class="typing-indicator"><span></span><span></span><span></span></span></div>
                </div>
            `;
            messagesDiv.appendChild(assistantDiv);
            const responseContent = document.getElementById('response-content');
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt: prompt,
                        temperature: parseFloat(tempSlider.value),
                        max_tokens: parseInt(tokensSlider.value)
                    })
                });
                
                const data = await response.json();
                responseContent.innerHTML = formatContent(data.response);
            } catch (error) {
                responseContent.innerHTML = `<span style="color: var(--error);">Error: ${error.message}</span>`;
            }
            
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
    </script>
</body>
</html>"""

if __name__ == "__main__":
    print("=" * 60)
    print("BLITZKODE SERVER")
    print("=" * 60)
    print("\n[URL] http://localhost:7860")
    print("[Stop] Ctrl+C\n")
    
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
