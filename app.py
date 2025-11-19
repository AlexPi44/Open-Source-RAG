import signal
import os
import shutil
import tempfile
import re
from pathlib import Path
import gradio as gr

# --- 🩹 CRITICAL FIX: Disable Signals for Gradio Threads ---
def no_op(*args, **kwargs): pass
signal.signal = no_op
signal.alarm = no_op

# --- ⚡ PERFORMANCE CONFIGURATION ---
# 1. Define a local folder for the model so the subprocess can GUARANTEE finding it
MODEL_CACHE_DIR = Path("./local_models").resolve()
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_PATH = str(MODEL_CACHE_DIR / "all-MiniLM-L6-v2")

# 2. Define LLM
LLM_MODEL = "Qwen/Qwen3-0.6B"

# --- BOOTSTRAP: FORCE OFFLINE LOADING ---
print("⏳ BOOT: Ensuring models are downloaded locally...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer

    # Download Embedding Model to specific folder
    if not os.path.exists(EMBED_MODEL_PATH):
        print(f"   Downloading {EMBED_MODEL_NAME} to {EMBED_MODEL_PATH}...")
        model = SentenceTransformer(EMBED_MODEL_NAME)
        model.save(EMBED_MODEL_PATH)
        print("   ✅ Embedding model saved locally.")
    else:
        print("   ✅ Embedding model found locally.")

    # Download LLM (Standard Cache is fine for LLM as it runs in main process)
    AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(LLM_MODEL, trust_remote_code=True)
    
    print("✅ BOOT: System Ready.")
except Exception as e:
    print(f"⚠️ BOOT WARNING: {e}")

# Now import LEANN
from leann import LeannBuilder, LeannChat
from pypdf import PdfReader

# --- APP CONFIG ---
INDEX_DIR = Path(tempfile.gettempdir()) / "leann_index"
INDEX_PATH_PREFIX = str(INDEX_DIR / "demo")

def clean_thinking_tags(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def build_index(files, openai_key):
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    
    if not files:
        return "Please upload at least one file.", None

    # Clear previous index
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize Builder with LOCAL PATH and SMALLER CHUNKS
        # chunk_size=256 is crucial for CPU performance with 30MB+ PDFs
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model=EMBED_MODEL_PATH, # <--- Passing path forces offline mode
            embedding_mode="sentence-transformers",
            chunk_size=256,  # <--- Smaller chunks = Faster Search
            chunk_overlap=20
        )
        
        doc_count = 0
        total_pages = 0
        
        for file in files:
            text = ""
            try:
                if file.name.endswith(".pdf"):
                    reader = PdfReader(file.name)
                    total_pages += len(reader.pages)
                    for page in reader.pages: 
                        text += page.extract_text() + "\n"
                else:
                    with open(file.name, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                
                if text.strip():
                    builder.add_text(text)
                    doc_count += 1
            except Exception as e:
                print(f"Error reading {file.name}: {e}")
                continue

        if doc_count == 0:
            return "No valid text extracted.", None

        builder.build_index(INDEX_PATH_PREFIX)
        
        return f"✅ Indexed {doc_count} file(s) ({total_pages} pages) using LEANN HNSW backend.", "ready"
        
    except Exception as e:
        return f"❌ Error building index: {str(e)}", None

def chat_response(message, history, mode, openai_key):
    if not INDEX_DIR.exists() or not any(INDEX_DIR.iterdir()):
        yield "⚠️ LEANN Index not found. Please build the index first."
        return

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    if mode == "Qwen3 (Free)":
        llm_config = {
            "type": "hf", 
            "model": LLM_MODEL,
            "trust_remote_code": True
        }
    else:
        if not openai_key:
            yield "⚠️ OpenAI Key required."
            return
        llm_config = {"type": "openai", "model": "gpt-4o-mini"}

    try:
        # LEANN Chat
        chat = LeannChat(INDEX_PATH_PREFIX, llm_config=llm_config)
        
        # Reduced top_k to 2 for speed on Free Tier
        raw_response = chat.ask(message, top_k=2)
        
        if mode == "Qwen3 (Free)":
            clean_response = clean_thinking_tags(raw_response)
            yield clean_response if clean_response else raw_response
        else:
            yield raw_response
            
    except Exception as e:
        yield f"Error: {str(e)}"

# --- UI ---
with gr.Blocks(title="LEANN RAG") as demo:
    gr.Markdown(f"# 🧠 LEANN RAG + {LLM_MODEL}")
    gr.Markdown("""
    **Description:** This tool uses **LEANN** (Lightweight Embedding & Neural Network) for highly efficient vector search.
    It compresses the index by 97% compared to traditional DBs, running entirely on this CPU-based Space.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            mode_select = gr.Dropdown(
                choices=["Qwen3 (Free)", "OpenAI (Fast)"], 
                value="Qwen3 (Free)", 
                label="LEANN Backend & Model"
            )
            api_key_input = gr.Textbox(
                label="OpenAI API Key", type="password", visible=False
            )
            
            def update_visibility(mode):
                return gr.Textbox(visible=(mode == "OpenAI (Fast)"))
            
            mode_select.change(update_visibility, mode_select, api_key_input)

            file_input = gr.File(label="Upload Documents (PDF/TXT)", file_count="multiple")
            build_btn = gr.Button("Build LEANN Index", variant="primary")
            build_status = gr.Textbox(label="System Status")

        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=chat_response,
                additional_inputs=[mode_select, api_key_input],
                type="messages",
                description="Ask questions about your documents. Powered by LEANN Vector Search."
            )

    build_btn.click(build_index, [file_input, api_key_input], [build_status])

if __name__ == "__main__":
    # Use 1 thread to prevent resource fighting on free tier
    demo.launch(max_threads=1)
