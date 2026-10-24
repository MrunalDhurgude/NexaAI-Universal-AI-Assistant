"""
app.py - NexaAI: Smart NLP & General-Purpose AI Chatbot.

Preserves the exact NexaAI dashboard UI design unchanged while delivering full
general-purpose AI chatbot capabilities:
- Python programming, code generation, line-by-line explanation, and debugging
- General knowledge, computer science, mathematics, and engineering concepts
- Multi-turn conversation memory with follow-up question support
- Dynamic Prompt Engineering: Zero-Shot, Role-Based, Few-Shot, Structured-Output
- Live Gemini API connection detection and robust error handling
"""

import os
import io
import time
import json
import hashlib
import datetime
import pandas as pd
import streamlit as st

# Local module imports
from prompts import (
    ALL_TECHNIQUES,
    TASK_QA,
    TASK_CODE,
    TASK_SUMMARIZATION,
    TASK_SENTIMENT,
    TASK_DISPLAY_NAMES,
    build_prompt,
    get_technique_description,
    TECHNIQUE_ZERO_SHOT,
    TECHNIQUE_ROLE_BASED,
    TECHNIQUE_FEW_SHOT,
    TECHNIQUE_STRUCTURED_OUTPUT,
)
from llm_service import LLMService, DEFAULT_MODELS, list_available_models
from dataset_utils import (
    load_dataset,
    get_dataset_summary,
    detect_text_columns,
    validate_evaluation_dataset,
    export_dataframe_to_csv,
)
from evaluation import (
    EVALUATION_RUBRIC,
    extract_sentiment_label,
    compute_compression_ratio,
    compute_accuracy,
    evaluate_consistency,
    plot_latency_comparison,
    plot_rubric_scores,
    plot_classification_distribution,
    export_records_to_csv,
)


def _get_available_models(api_key):
    if not api_key:
        return []
    key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    cached = st.session_state.get("gemini_model_cache")
    if cached and cached["key"] == key_fingerprint and time.monotonic() - cached["checked_at"] < 300:
        return cached["models"]

    models = list_available_models(api_key)
    st.session_state.gemini_model_cache = {
        "key": key_fingerprint,
        "models": models,
        "checked_at": time.monotonic(),
    }
    return models


def _get_connection_status(service, api_key, model_name):
    key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest() if api_key else ""
    cached = st.session_state.get("gemini_connection_cache")
    if (
        cached
        and cached["key"] == key_fingerprint
        and cached["model"] == model_name
        and time.monotonic() - cached["checked_at"] < 60
    ):
        service.connection_message = cached["message"]
        return cached["connected"], cached["label"]

    connected, label = service.test_connection()
    st.session_state.gemini_connection_cache = {
        "key": key_fingerprint,
        "model": model_name,
        "connected": connected,
        "label": label,
        "message": service.connection_message,
        "checked_at": time.monotonic(),
    }
    return connected, label

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="NexaAI | Smart NLP Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. EXACT DASHBOARD CSS STYLING (PRESERVED UNCHANGED)
# -----------------------------------------------------------------------------

st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Overall background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Sidebar Styling (Deep Navy/Indigo gradient with wave aesthetic) */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090e1a 0%, #0d1527 50%, #151a36 80%, #201a45 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 10px;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* Sidebar Radio Navigation items styled as modern SaaS pill buttons */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 8px;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #94a3b8 !important;
        font-weight: 500;
        font-size: 14px;
        padding: 10px 14px;
        border-radius: 10px;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        cursor: pointer;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.06);
    }
    
    /* Top Bar Status Pill */
    .top-status-bar {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 14px;
        padding: 4px 0 16px 0;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 6px 14px;
        border-radius: 9999px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        font-size: 13px;
        font-weight: 600;
        color: #1e293b;
    }
    .status-dot-green {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
        animation: pulse 2s infinite;
    }
    .status-dot-amber {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #f59e0b;
        box-shadow: 0 0 8px rgba(245, 158, 11, 0.6);
    }
    .status-dot-red {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #ef4444;
        box-shadow: 0 0 8px rgba(239, 68, 68, 0.6);
    }
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.15); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }

    /* Main Welcome Banner */
    .main-hero {
        margin-bottom: 20px;
    }
    .main-hero h1 {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.5px;
    }
    .main-hero p {
        font-size: 14px;
        color: #64748b;
        margin: 0;
    }

    /* 4 Feature Cards Grid */
    .feature-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 16px;
        transition: all 0.25s ease;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        border-color: #cbd5e1;
    }
    .feature-icon-wrapper {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        margin-bottom: 12px;
    }
    .feature-card h3 {
        font-size: 15px;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 6px 0;
    }
    .feature-card p {
        font-size: 12px;
        color: #64748b;
        line-height: 1.45;
        margin: 0;
    }

    /* Try These Examples Section */
    .section-title {
        font-size: 14px;
        font-weight: 700;
        color: #0f172a;
        margin: 22px 0 12px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Chat Container & Bubbles */
    .chat-box-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        margin-top: 18px;
        min-height: 380px;
    }

    /* User Message Bubble */
    .user-bubble-container {
        display: flex;
        justify-content: flex-end;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 16px;
    }
    .user-bubble {
        background-color: #dbeafe;
        color: #1e3a8a;
        padding: 12px 18px;
        border-radius: 16px 16px 4px 16px;
        font-size: 14px;
        line-height: 1.5;
        max-width: 78%;
        font-weight: 500;
    }
    .user-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background-color: #3b82f6;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 17px;
        flex-shrink: 0;
    }

    /* Assistant Message Bubble */
    .bot-bubble-container {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 20px;
    }
    .bot-avatar {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        color: #0284c7;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        flex-shrink: 0;
        border: 1px solid #93c5fd;
    }
    .bot-bubble {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #1e293b;
        padding: 16px 20px;
        border-radius: 16px 16px 16px 4px;
        font-size: 14px;
        line-height: 1.6;
        max-width: 88%;
    }

    .msg-timestamp {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 6px;
    }

    /* Right Panel Container */
    .right-panel-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    }
    .panel-header {
        font-size: 14px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .history-item {
        padding: 10px 12px;
        border-radius: 8px;
        background-color: #f8fafc;
        border: 1px solid #f1f5f9;
        margin-bottom: 8px;
        font-size: 13px;
        color: #334155;
        transition: all 0.2s;
    }
    .history-item:hover {
        background-color: #f1f5f9;
        border-color: #e2e8f0;
    }
    .history-meta {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Tags */
    .badge-tech {
        background: #e0f2fe;
        color: #0369a1;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. SESSION STATE INITIALIZATION & CONVERSATION MEMORY
# -----------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "user",
            "content": "What is Natural Language Processing?",
            "time": "10:24 AM",
            "metadata": {}
        },
        {
            "role": "assistant",
            "content": (
                "Natural Language Processing (NLP) is a branch of artificial intelligence "
                "that helps computers understand, interpret, and generate human language. "
                "It combines techniques from linguistics, machine learning, and deep learning "
                "to process text and speech. For example, NLP powers chatbots, machine translation, "
                "sentiment analysis, and automatic text summarization."
            ),
            "time": "10:24 AM",
            "metadata": {
                "task": TASK_QA,
                "technique": TECHNIQUE_ZERO_SHOT,
                "latency_sec": 0.42,
                "words": 52,
                "system_instruction": "You are NexaAI, a knowledgeable and professional general-purpose AI assistant.",
                "user_prompt": "What is Natural Language Processing?"
            }
        }
    ]

if "evaluation_history" not in st.session_state:
    st.session_state.evaluation_history = []

if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = {}

if "recent_queries" not in st.session_state:
    st.session_state.recent_queries = [
        {"query": "What is Natural Language Processing?", "time": "10:24 AM"},
        {"query": "Summarize climate change in 2 lines", "time": "10:18 AM"},
        {"query": "Is this product good?", "time": "10:12 AM"}
    ]

if "pending_input" not in st.session_state:
    st.session_state.pending_input = ""

if "learning_level" not in st.session_state:
    st.session_state.learning_level = "Intermediate"

if "dataset_cache" not in st.session_state:
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    if os.path.exists(dataset_path):
        try:
            st.session_state.dataset_cache = pd.read_csv(dataset_path)
        except Exception:
            st.session_state.dataset_cache = pd.DataFrame()
    else:
        st.session_state.dataset_cache = pd.DataFrame()

if "api_call_stats" not in st.session_state:
    st.session_state.api_call_stats = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_latency": 0.0,
        "history": []
    }


def _record_api_call(resp):
    """Safely logs non-sensitive request metrics for empirical evaluation."""
    if "api_call_stats" in st.session_state:
        stats = st.session_state.api_call_stats
        stats["total_requests"] += 1
        if resp.get("is_error"):
            stats["failed_requests"] += 1
        else:
            stats["successful_requests"] += 1
            stats["total_latency"] += float(resp.get("latency_sec", 0.0))
        stats["history"].insert(0, {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "model": resp.get("model_used", "N/A"),
            "latency_sec": resp.get("latency_sec", 0.0),
            "status": "Failed" if resp.get("is_error") else "Success",
            "error_type": resp.get("error_type") or "None"
        })
        if len(stats["history"]) > 50:
            stats["history"].pop()

# -----------------------------------------------------------------------------
# 4. SIDEBAR (NexaAI Branding, Navigation & API Configuration)
# -----------------------------------------------------------------------------

with st.sidebar:
    # Brand Header matching image exactly
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px; padding: 4px 6px;">
        <div style="width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #38bdf8 0%, #2563eb 100%); display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);">
            🤖
        </div>
        <div>
            <div style="font-size: 20px; font-weight: 800; color: #ffffff; letter-spacing: -0.4px;">
                Nexa<span style="color: #38bdf8;">AI</span>
            </div>
            <div style="font-size: 11px; color: #94a3b8; font-weight: 500;">
                Smart NLP Chatbot
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation Menu
    nav_selection = st.radio(
        label="Navigation",
        options=[
            "💬 Chatbot",
            "⚖️ Prompt Comparison",
            "🗄️ Dataset Testing",
            "📊 Evaluation",
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # API Key & Settings Expander (auto-expanded if key is missing)
    raw_key = os.getenv("GEMINI_API_KEY", "").strip()
    env_api_key = "" if raw_key.startswith("your_") else raw_key

    with st.expander("⚙️ Gemini API & Settings", expanded=not bool(env_api_key)):
        user_api_key = st.text_input(
            "Gemini API Key",
            value=env_api_key,
            type="password",
            placeholder="AIzaSy...",
            help="Get your key at https://aistudio.google.com/"
        )
        available_models = _get_available_models(user_api_key.strip())
        model_options = available_models or DEFAULT_MODELS
        configured_model = os.getenv("GEMINI_MODEL", "").strip()
        default_model = configured_model if configured_model in model_options else DEFAULT_MODELS[0]
        selected_model = st.selectbox(
            "Model",
            options=model_options,
            index=model_options.index(default_model) if default_model in model_options else 0,
            help="Models are loaded from the Gemini API for the configured key."
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Lower is more deterministic (ideal for coding/Q&A)"
        )
        learning_level = st.selectbox(
            "Learning Level",
            options=["Beginner", "Intermediate", "Advanced"],
            index=1,
            help="Adapts depth of explanations and examples"
        )
        st.session_state.learning_level = learning_level

        if st.button("🔌 Test Connection", use_container_width=True, key="btn_test_conn_sidebar"):
            with st.spinner("Testing Gemini connection..."):
                st.session_state.gemini_connection_cache = None
                t_srv = LLMService(api_key=user_api_key.strip(), model_name=selected_model)
                t_ok, t_lbl = t_srv.test_connection()
                if t_ok:
                    st.success(f"✔️ Connected ({selected_model})")
                else:
                    st.error(f"❌ {t_lbl}: {t_srv.connection_message}")

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        st.markdown("[🔗 Get Free Gemini API Key](https://aistudio.google.com/app/apikey)")

    # Sidebar Footer (AIDS Engineering Accreditation)
    st.markdown("""
    <div style="position: fixed; bottom: 20px; left: 16px; width: 230px;">
        <hr style="border-color: rgba(255,255,255,0.1); margin-bottom: 14px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="font-size: 22px;">🎓</div>
            <div>
                <div style="font-size: 12px; font-weight: 700; color: #e2e8f0;">AIDS Engineering</div>
                <div style="font-size: 11px; color: #94a3b8;">5th Semester NLP Micro Project</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initialize LLM Service instance
active_key = user_api_key.strip() if 'user_api_key' in locals() else os.getenv("GEMINI_API_KEY", "").strip()
active_model = selected_model if 'selected_model' in locals() else DEFAULT_MODELS[0]
llm_service = LLMService(api_key=active_key, model_name=active_model)

is_connected, conn_label = _get_connection_status(llm_service, active_key, active_model)


# =============================================================================
# VIEW 1: CHATBOT (EXACT MATCH TO EXISTING DASHBOARD DESIGN)
# =============================================================================
if nav_selection == "💬 Chatbot":
    # 1. Top Bar with Live Connection Status Indicator
    col_empty, col_status = st.columns([3, 1.5])
    with col_status:
        # Determine dot class and label
        if is_connected:
            dot_class = "status-dot-green"
            status_text = "Connected"
            status_color = "#10b981"
        elif conn_label == "Key Missing":
            dot_class = "status-dot-amber"
            status_text = "Key Missing"
            status_color = "#f59e0b"
        elif conn_label == "Quota Exceeded":
            dot_class = "status-dot-amber"
            status_text = "Quota Limit"
            status_color = "#f59e0b"
        else:
            dot_class = "status-dot-red"
            status_text = conn_label
            status_color = "#ef4444"

        st.markdown(f"""
        <div class="top-status-bar">
            <span style="font-size: 18px; cursor: pointer;">☀️</span>
            <div class="status-pill">
                <span style="font-size: 16px;">✨</span>
                <span>Gemini API</span>
                <span class="{dot_class}"></span>
                <span style="color: {status_color}; font-size: 12px;">{status_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if not is_connected and llm_service.connection_message:
            st.caption(llm_service.connection_message)

    # 2. Main Two-Column Layout (Chat Area 72% | Controls & History 28%)
    main_col, side_col = st.columns([2.5, 1], gap="medium")

    with main_col:
        # Hero Title Header (Preserved Exactly)
        st.markdown("""
        <div class="main-hero">
            <h1><span style="color: #3b82f6;">✨</span> Your Intelligent AI Assistant</h1>
            <p>Ask questions, get summaries, analyze sentiment, and more — powered by Gemini.</p>
        </div>
        """, unsafe_allow_html=True)

        # 4 Quick Action Feature Cards in a 4-column row (Preserved Exactly)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-wrapper" style="background-color: #e0f2fe; color: #0284c7;">💬</div>
                <h3>AI Chat</h3>
                <p>Ask anything and get helpful, accurate answers.</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-wrapper" style="background-color: #dcfce7; color: #16a34a;">📄</div>
                <h3>Summarization</h3>
                <p>Turn long text into short, meaningful points.</p>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-wrapper" style="background-color: #fce7f3; color: #db2777;">💖</div>
                <h3>Sentiment Analysis</h3>
                <p>Classify text as Positive, Negative or Neutral.</p>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-wrapper" style="background-color: #f3e8ff; color: #9333ea;">⚙️</div>
                <h3>Prompt Engineering</h3>
                <p>Use different prompt techniques for better results.</p>
            </div>
            """, unsafe_allow_html=True)

        # "Try These Examples" Section (Preserved Exactly)
        st.markdown("<div class='section-title'>💡 Try These Examples</div>", unsafe_allow_html=True)
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            if st.button("💬 Explain Artificial Intelligence", use_container_width=True):
                st.session_state.pending_input = "Explain Artificial Intelligence, machine learning, and deep learning in simple terms with an intuitive comparison."
                st.rerun()
        with ex2:
            if st.button("📄 Summarize this text", use_container_width=True):
                st.session_state.pending_input = "Artificial intelligence is revolutionizing healthcare by assisting radiologists in detecting early-stage tumors from MRI scans with superhuman speed. Hospitals adopting automated diagnostic pipelines report a 35% reduction in diagnostic backlog. However, regulatory authorities insist that human physicians must retain ultimate decision-making authority to avoid liability."
                st.rerun()
        with ex3:
            if st.button("💖 Analyze sentiment", use_container_width=True):
                st.session_state.pending_input = "The build quality and audio clarity exceeded all my expectations. The battery lasted 30 hours straight on my flight!"
                st.rerun()

        # Conversation Card Container (Preserved Exactly)
        st.markdown("<div class='chat-box-card'>", unsafe_allow_html=True)

        if not st.session_state.chat_history:
            st.info("NexaAI conversation history is clear. Enter a query or select an example above to begin!")

        for i, msg in enumerate(st.session_state.chat_history):
            msg_time = msg.get("time", "10:24 AM")
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="user-bubble-container">
                    <div style="text-align: right;">
                        <div class="user-bubble">{msg['content']}</div>
                        <div class="msg-timestamp">{msg_time}</div>
                    </div>
                    <div class="user-avatar">👤</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                meta = msg.get("metadata", {})
                tech_tag = meta.get("technique", "Zero-Shot")
                latency_val = meta.get("latency_sec", 0.4)

                st.markdown(f"""
                <div class="bot-bubble-container">
                    <div class="bot-avatar">🤖</div>
                    <div style="flex-grow: 1;">
                        <div class="bot-bubble">
                """, unsafe_allow_html=True)

                # Render markdown content including Python code blocks cleanly
                content_text = msg["content"]
                
                # Check if structured output JSON
                if tech_tag == TECHNIQUE_STRUCTURED_OUTPUT and "{" in content_text and "}" in content_text:
                    try:
                        clean_json = content_text.strip()
                        if clean_json.startswith("```json"):
                            clean_json = clean_json[7:]
                        if clean_json.startswith("```"):
                            clean_json = clean_json[3:]
                        if clean_json.endswith("```"):
                            clean_json = clean_json[:-3]
                        parsed_json = json.loads(clean_json.strip())
                        
                        # Display nicely formatted summary + interactive json
                        if isinstance(parsed_json, dict):
                            if "topic" in parsed_json:
                                st.markdown(f"### {parsed_json.get('topic')}")
                            if "summary" in parsed_json:
                                st.markdown(f"**Summary:** {parsed_json.get('summary')}")
                            if "detailed_explanation" in parsed_json:
                                st.markdown(parsed_json.get("detailed_explanation"))
                            
                            code_obj = parsed_json.get("code_solution", {})
                            if isinstance(code_obj, dict) and code_obj.get("code"):
                                st.markdown(f"#### Python Implementation")
                                st.code(code_obj["code"], language="python")
                                if code_obj.get("time_complexity"):
                                    st.caption(f"**Complexity:** Time: {code_obj.get('time_complexity')} | Space: {code_obj.get('space_complexity')}")
                            
                            with st.expander("🔍 View Raw Structured JSON Payload"):
                                st.json(parsed_json)
                        else:
                            st.markdown(content_text)
                    except Exception:
                        st.markdown(content_text)
                else:
                    st.markdown(content_text)

                st.markdown(f"""
                        </div>
                        <div class="msg-timestamp">{msg_time} &bull; <span class="badge-tech">{tech_tag}</span> &bull; ⏱️ {latency_val}s</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Star Rating & Feedback Action Bar (Preserved Exactly)
                col_r1, col_r2, col_r3 = st.columns([2, 1, 3])
                with col_r1:
                    st.caption("⭐ Rate this response: [ ★ ★ ★ ★ ★ ]")
                with col_r2:
                    with st.popover("💬 Feedback"):
                        with st.form(f"rate_form_{i}"):
                            st.write("Rate this response (1-5 Rubric):")
                            r_rel = st.slider("Relevance", 1, 5, 5, key=f"rf_rel_{i}", help=EVALUATION_RUBRIC["relevance"]["description"])
                            r_cor = st.slider("Correctness", 1, 5, 5, key=f"rf_cor_{i}", help=EVALUATION_RUBRIC["correctness"]["description"])
                            r_cla = st.slider("Clarity", 1, 5, 5, key=f"rf_cla_{i}", help=EVALUATION_RUBRIC["clarity"]["description"])
                            r_note = st.text_input("Reviewer Feedback Note", key=f"rf_note_{i}")
                            if st.form_submit_button("Save Feedback"):
                                st.session_state.evaluation_history.append({
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "task": meta.get("task", TASK_QA),
                                    "technique": meta.get("technique", TECHNIQUE_ZERO_SHOT),
                                    "input_text": msg.get("content", ""),
                                    "relevance": r_rel,
                                    "correctness": r_cor,
                                    "clarity": r_cla,
                                    "overall_score": round((r_rel + r_cor + r_cla)/3, 2),
                                    "latency_sec": latency_val,
                                    "words": len(content_text.split()),
                                    "notes": r_note
                                })
                                st.success("Feedback saved to evaluation records!")

        st.markdown("</div>", unsafe_allow_html=True)

        # Bottom Input Area
        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        
        user_input = st.chat_input("Type your message...", key="user_chat_input")

        # If user clicked an example button
        if not user_input and st.session_state.pending_input:
            user_input = st.session_state.pending_input
            st.session_state.pending_input = ""

        # Bottom Toolbar
        t_col1, t_col2 = st.columns([1, 1])
        with t_col1:
            if st.button("🗑️ Clear Chat", key="btn_clear_chat_bottom"):
                st.session_state.chat_history = []
                st.session_state.pending_input = ""
                st.rerun()
        with t_col2:
            st.markdown("<div style='text-align: right; color: #94a3b8; font-size: 12px; padding-top: 8px;'>0/2000</div>", unsafe_allow_html=True)

    # 3. Right Sidebar Panel (Task Selector, Prompt Technique & History) - Preserved Exactly
    with side_col:
        st.markdown("<div class='right-panel-card'>", unsafe_allow_html=True)
        
        # 1. Select Task Dropdown
        st.markdown("<div class='panel-header'>🎯 Select Task</div>", unsafe_allow_html=True)
        task_choice = st.selectbox(
            "Task",
            options=[TASK_QA, TASK_CODE, TASK_SUMMARIZATION, TASK_SENTIMENT],
            format_func=lambda x: {
                TASK_QA: "General Q&A",
                TASK_CODE: "Code Generation & Debugging",
                TASK_SUMMARIZATION: "Text Summarization",
                TASK_SENTIMENT: "Sentiment Analysis"
            }.get(x, x),
            label_visibility="collapsed"
        )

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # 2. Select Prompt Technique Dropdown
        st.markdown("<div class='panel-header'>⚡ Select Prompt Technique</div>", unsafe_allow_html=True)
        tech_choice = st.selectbox(
            "Prompt Technique",
            options=ALL_TECHNIQUES,
            format_func=lambda x: {
                TECHNIQUE_ZERO_SHOT: "Zero-Shot (Basic)",
                TECHNIQUE_ROLE_BASED: "Role-Based (Persona)",
                TECHNIQUE_FEW_SHOT: "Few-Shot (Examples)",
                TECHNIQUE_STRUCTURED_OUTPUT: "Structured-Output (JSON)"
            }.get(x, x),
            label_visibility="collapsed"
        )

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # 3. Chat History List
        st.markdown("<div class='panel-header'>🕒 Chat History</div>", unsafe_allow_html=True)
        
        for q_item in st.session_state.recent_queries:
            st.markdown(f"""
            <div class="history-item">
                <div style="font-weight: 600;">{q_item['query']}</div>
                <div class="history-meta">&bull; {q_item['time']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='text-align: right; margin-top: 4px;'><a style='font-size: 12px; color: #3b82f6; text-decoration: none; font-weight: 600;' href='#'>View All &gt;</a></div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

        # 4. Clear Chat Wide Action Button
        if st.button("🔄 Clear Chat", use_container_width=True, key="btn_clear_chat_right"):
            st.session_state.chat_history = []
            st.session_state.pending_input = ""
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # 4. Execution Pipeline for Incoming User Input
    if user_input:
        cur_time = datetime.datetime.now().strftime("%I:%M %p")
        
        # 1. Append user message to conversation memory
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "time": cur_time,
            "metadata": {}
        })

        # 2. Update recent queries in history list
        st.session_state.recent_queries.insert(0, {
            "query": user_input[:38] + ("..." if len(user_input) > 38 else ""),
            "time": cur_time
        })
        if len(st.session_state.recent_queries) > 5:
            st.session_state.recent_queries.pop()

        # 3. Build dynamic prompt with user level & prompt technique
        sys_inst, usr_prompt = build_prompt(
            task_type=task_choice,
            technique=tech_choice,
            user_input=user_input,
            learning_level=st.session_state.learning_level
        )

        # 4. Invoke LLM with multi-turn conversation memory
        with st.spinner("NexaAI is generating response..."):
            resp = llm_service.generate_response(
                system_instruction=sys_inst,
                user_prompt=usr_prompt,
                conversation_history=st.session_state.chat_history[:-1],  # prior turns for memory
                temperature=temperature if 'temperature' in locals() else 0.2
            )
            _record_api_call(resp)

        # 5. Append assistant response to conversation memory
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": resp["text"],
            "time": datetime.datetime.now().strftime("%I:%M %p"),
            "metadata": {
                "task": task_choice,
                "technique": tech_choice,
                "latency_sec": resp["latency_sec"],
                "words": len(resp["text"].split()),
                "system_instruction": sys_inst,
                "user_prompt": usr_prompt
            }
        })
        st.rerun()


# =============================================================================
# VIEW 2: PROMPT COMPARISON MATRIX (PRESERVED)
# =============================================================================
elif nav_selection == "⚖️ Prompt Comparison":
    st.markdown("""
    <div class="main-hero">
        <h1><span style="color: #3b82f6;">⚖️</span> Multi-Technique Prompt Comparison</h1>
        <p>Run any query or coding problem through Zero-Shot, Role-Based, Few-Shot, and Structured-Output prompts simultaneously.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        cmp_task = st.selectbox(
            "Task Domain",
            options=[TASK_QA, TASK_SUMMARIZATION, TASK_SENTIMENT],
            format_func=lambda x: {
                TASK_QA: "General Q&A / Python / Engineering",
                TASK_SUMMARIZATION: "Text Summarization",
                TASK_SENTIMENT: "Sentiment Analysis"
            }.get(x, x),
            key="cmp_task"
        )
    with col2:
        selected_techniques = st.multiselect(
            "Select Techniques to Compare",
            options=ALL_TECHNIQUES,
            default=ALL_TECHNIQUES,
            key="cmp_techniques"
        )

    # Sample loader from dataset or custom
    df_ds = st.session_state.dataset_cache
    cmp_default_text = ""
    if not df_ds.empty and "task_type" in df_ds.columns:
        matching = df_ds[df_ds["task_type"] == cmp_task]
        if not matching.empty:
            sel_cmp_sample = st.selectbox(
                "Pick a benchmark sample (Optional):",
                options=["-- Custom Input --"] + matching["input_text"].tolist(),
                key="sel_cmp_sample"
            )
            if sel_cmp_sample != "-- Custom Input --":
                cmp_default_text = sel_cmp_sample

    input_text = st.text_area(
        "Input Query, Coding Problem, or Text:",
        value=cmp_default_text or "Write a Python function to check if a binary tree is balanced, and explain its time complexity.",
        height=110,
        placeholder="Enter any general query, algorithm problem, or paragraph..."
    )

    if st.button("🚀 Run Comparative Benchmark", type="primary", use_container_width=True):
        if not input_text.strip():
            st.error("Please enter input text or pick a sample.")
        else:
            comparison_results = {}
            progress_bar = st.progress(0)
            
            for i, tech in enumerate(selected_techniques):
                sys_inst, usr_prompt = build_prompt(
                    task_type=cmp_task,
                    technique=tech,
                    user_input=input_text,
                    learning_level=st.session_state.learning_level
                )
                
                resp = llm_service.generate_response(
                    system_instruction=sys_inst,
                    user_prompt=usr_prompt,
                    conversation_history=None,
                    temperature=0.2
                )
                _record_api_call(resp)
                
                comparison_results[tech] = {
                    "output": resp["text"],
                    "latency": resp["latency_sec"],
                    "words": len(resp["text"].split()),
                    "system_prompt": sys_inst,
                    "user_prompt": usr_prompt
                }
                progress_bar.progress((i + 1) / len(selected_techniques))

            st.session_state.comparison_results = comparison_results
            st.success("Comparison completed!")

    if st.session_state.comparison_results:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.comparison_results))

        for col, (tech, data) in zip(cols, st.session_state.comparison_results.items()):
            with col:
                st.markdown(f"""
                <div style="background: white; border-radius: 12px; border: 1px solid #e2e8f0; padding: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
                    <div style="font-weight: 700; color: #1e293b; font-size: 15px; margin-bottom: 6px;">{tech}</div>
                    <div style="font-size: 12px; color: #64748b; margin-bottom: 12px;">⏱️ Latency: <b>{data['latency']}s</b> | 📝 Words: <b>{data['words']}</b></div>
                </div>
                """, unsafe_allow_html=True)

                if tech == TECHNIQUE_STRUCTURED_OUTPUT:
                    try:
                        clean_json = data["output"].strip()
                        if clean_json.startswith("```json"):
                            clean_json = clean_json[7:]
                        if clean_json.startswith("```"):
                            clean_json = clean_json[3:]
                        if clean_json.endswith("```"):
                            clean_json = clean_json[:-3]
                        st.json(json.loads(clean_json.strip()))
                    except Exception:
                        st.markdown(data["output"])
                else:
                    st.markdown(data["output"])

                with st.expander("🔍 View Prompt Payload"):
                    st.caption("**System Instruction:**")
                    st.code(data["system_prompt"], language="text")
                    st.caption("**User Prompt:**")
                    st.code(data["user_prompt"], language="text")


# =============================================================================
# VIEW 3: DATASET TESTING (PRESERVED & ENHANCED)
# =============================================================================
elif nav_selection == "🗄️ Dataset Testing":
    st.markdown("""
    <div class="main-hero">
        <h1><span style="color: #3b82f6;">🗄️</span> Dataset Testing & Automated Accuracy</h1>
        <p>Benchmark standardized NLP test cases or upload a custom CSV dataset. Profile dataset statistics and run real model evaluations.</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Dataset Source Selector (Upload vs Built-in)
    col_src1, col_src2 = st.columns([1.5, 2])
    with col_src1:
        ds_source = st.radio(
            "Select Data Source",
            options=["Standard NLP Benchmark (dataset.csv)", "Upload Custom CSV File"],
            horizontal=True
        )

    active_df = pd.DataFrame()
    if ds_source == "Upload Custom CSV File":
        uploaded_csv = st.file_uploader("Upload CSV File for Evaluation", type=["csv"], help="Upload any standard CSV file with text columns")
        if uploaded_csv is not None:
            try:
                active_df = load_dataset(uploaded_csv)
                st.success(f"Custom CSV loaded: {len(active_df)} rows, {len(active_df.columns)} columns")
            except Exception as e:
                st.error(f"Failed to load custom CSV: {e}")
                active_df = st.session_state.dataset_cache
        else:
            st.info("Upload a CSV file above or switch to the standard benchmark.")
            active_df = st.session_state.dataset_cache
    else:
        active_df = st.session_state.dataset_cache

    if active_df.empty:
        st.warning("No dataset loaded. Please upload a CSV file or verify that dataset.csv exists in the project directory.")
    else:
        # 2. Dataset Statistical Profiling
        summary = get_dataset_summary(active_df)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Row Count", summary["num_rows"])
        p2.metric("Column Count", summary["num_columns"])
        total_missing = sum(summary["missing_values"].values())
        p3.metric("Missing Values", total_missing)
        p4.metric("Memory Footprint", f"{summary['memory_kb']} KB")

        with st.expander("🔍 Dataset Profiling & Missing Values Breakdown"):
            c_prof1, c_prof2 = st.columns(2)
            with c_prof1:
                st.markdown("**Missing Values per Column:**")
                st.json(summary["missing_values"])
            with c_prof2:
                st.markdown("**Column Data Types:**")
                st.json(summary["dtypes"])

        st.markdown("##### Dataset Preview")
        st.dataframe(active_df.head(10), use_container_width=True)

        # 3. Column Selection and Test Configuration
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("##### ⚙️ Evaluation Configuration")

        col_c1, col_c2, col_c3 = st.columns(3)
        detected_text_cols = detect_text_columns(active_df) or list(active_df.columns)

        with col_c1:
            selected_text_col = st.selectbox(
                "Text Column to Evaluate",
                options=detected_text_cols,
                index=0,
                help="Column containing questions, paragraphs, or customer feedback."
            )

        with col_c2:
            batch_task = st.selectbox(
                "Task Domain",
                options=[TASK_QA, TASK_CODE, TASK_SUMMARIZATION, TASK_SENTIMENT],
                format_func=lambda x: {
                    TASK_QA: "General Q&A",
                    TASK_CODE: "Code Generation & Debugging",
                    TASK_SUMMARIZATION: "Text Summarization",
                    TASK_SENTIMENT: "Sentiment Analysis"
                }.get(x, x),
                key="ds_batch_task"
            )

        with col_c3:
            batch_tech = st.selectbox("Prompt Technique", options=ALL_TECHNIQUES, key="ds_batch_tech")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            potential_labels = ["-- None (No Ground-Truth) --"] + [c for c in active_df.columns if c != selected_text_col]
            def_lbl_idx = potential_labels.index("expected_output") if "expected_output" in potential_labels else 0
            selected_label_col = st.selectbox(
                "Expected Ground-Truth Column (Optional)",
                options=potential_labels,
                index=def_lbl_idx,
                help="Select if dataset contains verified target answers/labels for accuracy scoring."
            )
        with col_s2:
            max_samples = st.slider("Samples to Benchmark", 1, min(len(active_df), 20), min(len(active_df), 5))

        # 4. Run Batch Evaluation Pipeline
        if st.button("🚀 Run Batch Evaluation Pipeline", type="primary", use_container_width=True):
            eval_subset = active_df.head(max_samples).copy()
            batch_records = []
            progress_bar = st.progress(0)

            for i, (_, row) in enumerate(eval_subset.iterrows()):
                inp_text = str(row[selected_text_col]).strip()
                expected_str = ""
                if selected_label_col != "-- None (No Ground-Truth) --":
                    expected_str = str(row[selected_label_col]).strip()

                sys_inst, usr_prompt = build_prompt(
                    task_type=batch_task,
                    technique=batch_tech,
                    user_input=inp_text,
                    learning_level=st.session_state.learning_level
                )

                resp = llm_service.generate_response(
                    system_instruction=sys_inst,
                    user_prompt=usr_prompt,
                    conversation_history=None,
                    temperature=0.2
                )
                _record_api_call(resp)

                model_out = resp["text"]
                pred_label = "N/A"
                is_correct = "N/A"

                if batch_task == TASK_SENTIMENT:
                    pred_label = extract_sentiment_label(model_out)
                    if expected_str:
                        is_correct = "✅ Match" if pred_label.lower() == expected_str.lower() else "❌ Mismatch"

                batch_records.append({
                    "Sample #": i + 1,
                    "Input Text": inp_text[:75] + ("..." if len(inp_text) > 75 else ""),
                    "Expected Target": expected_str if expected_str else "N/A",
                    "Model Output": pred_label if batch_task == TASK_SENTIMENT else (model_out[:120] + "..."),
                    "Match Status": is_correct,
                    "Latency (s)": resp["latency_sec"],
                    "Technique": batch_tech
                })

                progress_bar.progress((i + 1) / len(eval_subset))

            batch_df = pd.DataFrame(batch_records)

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)

            if batch_task == TASK_SENTIMENT and selected_label_col != "-- None (No Ground-Truth) --":
                matches = sum(1 for r in batch_records if r["Match Status"] == "✅ Match")
                acc_pct = round((matches / len(batch_records)) * 100, 2)
                m1.metric("Classification Accuracy", f"{acc_pct}%", f"{matches}/{len(batch_records)} match")
            else:
                m1.metric("Evaluated Test Examples", len(batch_df))

            m2.metric("Mean Latency", f"{batch_df['Latency (s)'].mean():.2f}s")
            m3.metric("Selected Technique", batch_tech)

            st.dataframe(batch_df, use_container_width=True)

            csv_data = export_dataframe_to_csv(batch_df)
            st.download_button(
                "📥 Download Batch Results as CSV",
                data=csv_data,
                file_name=f"dataset_results_{batch_tech.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )


# =============================================================================
# VIEW 4: EVALUATION & ANALYTICS DASHBOARD (PRESERVED & ENHANCED)
# =============================================================================
elif nav_selection == "📊 Evaluation":
    st.markdown("""
    <div class="main-hero">
        <h1><span style="color: #3b82f6;">📊</span> Evaluation & Analytics Dashboard</h1>
        <p>Empirical metrics: Human evaluation ratings, latency benchmarks, consistency checks, and academic project reports.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_metrics, tab_consistency, tab_academic = st.tabs([
        "📈 Empirical Analytics & Charts",
        "🔄 Query Consistency Tester",
        "📚 Academic Report & Viva Voce"
    ])

    with tab_metrics:
        stats = st.session_state.api_call_stats
        eval_history = st.session_state.evaluation_history
        total_calls = stats["total_requests"]
        succ_calls = stats["successful_requests"]
        fail_calls = stats["failed_requests"]
        avg_lat = round(stats["total_latency"] / max(1, succ_calls), 2)

        # Top Session KPI Metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Test Examples", total_calls)
        k2.metric("Successful Responses", succ_calls)
        k3.metric("Failed Requests", fail_calls)
        k4.metric("Avg Response Time", f"{avg_lat}s")

        if total_calls == 0 and not eval_history:
            st.info(
                "ℹ️ **No empirical test data logged yet.**\n\n"
                "Run prompts in **💬 Chatbot**, test techniques in **⚖️ Prompt Comparison**, "
                "or execute evaluations in **🗄️ Dataset Testing** to record live performance statistics."
            )
        else:
            if stats["history"]:
                st.markdown("##### ⏱️ Live Session API Call Log")
                st.dataframe(pd.DataFrame(stats["history"]), use_container_width=True)

            if eval_history:
                df_eval = pd.DataFrame(eval_history)
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                st.markdown("##### 🏆 Human Evaluation Rubric (Relevance, Correctness, Clarity)")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Total Human Ratings", len(df_eval))
                r2.metric("Avg Relevance", f"{df_eval['relevance'].mean():.2f} / 5.0")
                r3.metric("Avg Correctness", f"{df_eval['correctness'].mean():.2f} / 5.0")
                r4.metric("Avg Clarity", f"{df_eval['clarity'].mean():.2f} / 5.0")

                c_fig1, c_fig2 = st.columns(2)
                with c_fig1:
                    st.markdown("###### Latency Benchmark Across Techniques")
                    fig_lat = plot_latency_comparison(df_eval)
                    st.pyplot(fig_lat)

                with c_fig2:
                    st.markdown("###### Rubric Dimension Distribution")
                    fig_rub = plot_rubric_scores(df_eval)
                    st.pyplot(fig_rub)

                st.markdown("##### Logged Human Evaluation Records")
                st.dataframe(df_eval, use_container_width=True)

                c_dl, c_cl = st.columns([2, 1])
                with c_dl:
                    st.download_button(
                        "📥 Download Evaluation History (CSV)",
                        data=export_records_to_csv(eval_history),
                        file_name="nexaai_evaluations.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with c_cl:
                    if st.button("🗑️ Clear Evaluation History", use_container_width=True):
                        st.session_state.evaluation_history = []
                        st.session_state.api_call_stats = {
                            "total_requests": 0,
                            "successful_requests": 0,
                            "failed_requests": 0,
                            "total_latency": 0.0,
                            "history": []
                        }
                        st.rerun()

    with tab_consistency:
        st.markdown("#### Semantic Query Consistency & Robustness Tester")
        st.caption("Tests whether the chosen prompt technique produces identical or consistent outputs across 3 paraphrased queries.")

        c_task = st.selectbox("Consistency Task", [TASK_SENTIMENT, TASK_QA, TASK_CODE])
        c_tech = st.selectbox("Technique", ALL_TECHNIQUES, key="c_tech_sel")

        if c_task == TASK_SENTIMENT:
            q1_d = "The battery lasts 15 hours and the performance is remarkably smooth."
            q2_d = "I am so impressed with how long the battery lasts and how fluid the speed is!"
            q3_d = "Exceptional 15-hour battery duration paired with seamless fast performance."
        elif c_task == TASK_CODE:
            q1_d = "Write a Python function to reverse a linked list."
            q2_d = "Provide Python code to reverse a singly linked list in-place."
            q3_d = "How do you reverse a singly linked list in Python with O(1) space?"
        else:
            q1_d = "What is the primary role of self-attention in Transformer models?"
            q2_d = "How does the self-attention mechanism work in Transformers?"
            q3_d = "Explain the function and significance of self-attention in Transformers."

        v1_col, v2_col, v3_col = st.columns(3)
        with v1_col:
            v1 = st.text_input("Variant 1", value=q1_d)
        with v2_col:
            v2 = st.text_input("Variant 2", value=q2_d)
        with v3_col:
            v3 = st.text_input("Variant 3", value=q3_d)

        if st.button("🧪 Execute Consistency Test", use_container_width=True):
            variants = [v1, v2, v3]
            variant_outputs = []

            with st.spinner("Analyzing variants..."):
                for v in variants:
                    s_inst, u_pr = build_prompt(c_task, c_tech, v, st.session_state.learning_level)
                    r = llm_service.generate_response(
                        system_instruction=s_inst,
                        user_prompt=u_pr,
                        conversation_history=None,
                        temperature=0.0
                    )
                    _record_api_call(r)
                    variant_outputs.append(r["text"])

            report = evaluate_consistency(variants, variant_outputs, c_task)

            st.metric("Consistency Alignment Score", f"{report['consistency_score']}%")
            st.info(f"Status: {report['status']}")
            for idx, (var, out) in enumerate(zip(variants, variant_outputs), 1):
                st.markdown(f"**Variant {idx}:** *\"{var}\"*")
                st.caption(f"**Output:** {out[:120]}...")

    with tab_academic:
        st.markdown("""
        ### 📄 Project Abstract & System Summary
        This project implements **NexaAI**, a functional, general-purpose AI chatbot powered by the **Google Gemini API**
        and developed in **Python** using **Streamlit**. NexaAI answers questions from beginner to advanced levels
        across programming, Python, computer science, mathematics, engineering, academics, and general knowledge.

        #### 💡 Core Viva Voce Questions & Answers
        1. **Q: How does NexaAI support multi-turn follow-up questions?**  
           *A:* NexaAI maintains conversation history in Streamlit session state and supplies a sliding context window of prior turns to the Google GenAI SDK, allowing users to ask follow-up questions without repeating previous context.
        2. **Q: How does NexaAI assist in Python programming?**  
           *A:* NexaAI explains problem requirements, provides formatted Python code with type hints, breaks down key lines, offers sample inputs/outputs, debugs errors, and provides time/space complexity analysis.
        3. **Q: What is the benefit of Structured-Output prompting?**  
           *A:* It enforces a deterministic JSON schema containing designated fields (`topic`, `summary`, `code_solution`, `time_complexity`), making outputs directly usable by downstream software APIs.
        4. **Q: How does the Gemini API status indicator work?**  
           *A:* It performs a live test connection check, dynamically indicating whether the API key is Connected (green), Key Missing (amber), or Quota Exceeded (amber).
        """)
