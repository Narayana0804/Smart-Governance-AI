"""
Smart Governance AI - Streamlit Dashboard
Premium Admin monitoring interface with real-time predictions, topic charts, and fake news detection.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time

# ── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Governance AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

* { 
    font-family: 'Outfit', sans-serif; 
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.5);
}
::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.5);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.8);
}

.stApp {
    background-color: #05050f;
    background-image: 
        radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
        radial-gradient(at 50% 0%, hsla(225,39%,30%,0.2) 0, transparent 50%), 
        radial-gradient(at 100% 0%, hsla(339,49%,30%,0.2) 0, transparent 50%);
    background-attachment: fixed;
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background-color: rgba(10, 10, 20, 0.6) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Glassmorphic Metrics */
div[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
    border-color: rgba(129, 140, 248, 0.5);
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.25);
    background: rgba(255, 255, 255, 0.04);
}

div[data-testid="metric-container"] > div {
    text-align: center;
}

div[data-testid="metric-container"] label {
    font-size: 1.05rem !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px;
}

div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #c084fc, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h1, h2, h3, h4, h5, h6 { 
    color: #f8fafc !important; 
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
}

.hero-title {
    font-size: 4rem;
    font-weight: 900;
    background: linear-gradient(100deg, #818cf8, #e879f9, #38bdf8);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
    line-height: 1.2;
    animation: gradientShift 5s ease infinite, fadeInDown 0.8s ease-out;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero-subtitle {
    color: #cbd5e1;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 400;
    margin-top: 0;
    margin-bottom: 2rem;
    animation: fadeInUp 0.8s ease-out 0.2s both;
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-card {
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 30px;
    margin: 15px 0;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
}

.result-card:hover {
    transform: translateY(-8px);
    border-color: rgba(99, 102, 241, 0.5);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 20px rgba(99, 102, 241, 0.1) inset;
    background: rgba(30, 41, 59, 0.5);
}

.result-card h3, .result-card h4 {
    margin-top: 0;
    font-weight: 700;
    color: #f1f5f9;
}

/* Add glows based on urgency */
.urgency-critical { box-shadow: 0 0 30px rgba(239, 68, 68, 0.15) inset; border-color: rgba(239, 68, 68, 0.4); }
.urgency-high { box-shadow: 0 0 30px rgba(249, 115, 22, 0.15) inset; border-color: rgba(249, 115, 22, 0.4); }
.urgency-medium { box-shadow: 0 0 30px rgba(234, 179, 8, 0.15) inset; border-color: rgba(234, 179, 8, 0.4); }
.urgency-low { box-shadow: 0 0 30px rgba(34, 197, 94, 0.15) inset; border-color: rgba(34, 197, 94, 0.4); }

.badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 6px 16px;
    border-radius: 30px;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.badge-critical { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.5); }
.badge-high { background: rgba(249, 115, 22, 0.2); color: #fdba74; border: 1px solid rgba(249, 115, 22, 0.5); }
.badge-medium { background: rgba(234, 179, 8, 0.2); color: #fde047; border: 1px solid rgba(234, 179, 8, 0.5); }
.badge-low { background: rgba(34, 197, 94, 0.2); color: #86efac; border: 1px solid rgba(34, 197, 94, 0.5); }
.badge-safe { background: rgba(34, 197, 94, 0.2); color: #86efac; border: 1px solid rgba(34, 197, 94, 0.5); }
.badge-suspicious { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.5); animation: pulse 2s infinite; }

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
    70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

.stat-number {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin: 10px 0;
}

div[data-testid="stTextArea"] textarea {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 20px !important;
    color: #f8fafc !important;
    padding: 24px !important;
    font-size: 1.1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.25), inset 0 2px 4px rgba(0,0,0,0.2) !important;
    background: rgba(30, 41, 59, 0.6) !important;
}

/* Beautiful Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 16px 36px !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39) !important;
    width: 100% !important;
    position: relative;
    overflow: hidden;
}

.stButton > button::after {
    content: '';
    position: absolute;
    top: 0; left: -100%; width: 50%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: all 0.5s ease;
}

.stButton > button:hover::after {
    left: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
}

.stButton > button:active {
    transform: translateY(1px) !important;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.4) !important;
}

/* DataFrame Styling */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

/* Block container spacing */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: rgba(15, 23, 42, 0.4);
    border-radius: 12px;
    padding: 5px;
    backdrop-filter: blur(10px);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #cbd5e1 !important;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 20px;
    margin: 0 2px;
    transition: all 0.3s;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background-color: rgba(99, 102, 241, 0.2) !important;
    color: #818cf8 !important;
}

</style>
""", unsafe_allow_html=True)


def call_api(endpoint: str, data: dict = None, method: str = "POST"):
    """Call FastAPI backend."""
    try:
        url = f"{API_URL}{endpoint}"
        if method == "POST":
            resp = requests.post(url, json=data, timeout=30)
        else:
            resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def get_urgency_badge(urgency: str) -> str:
    cls = urgency.lower()
    return f'<span class="badge badge-{cls}">{urgency}</span>'


def get_suspicion_badge(is_suspicious: bool) -> str:
    if is_suspicious:
        return '<span class="badge badge-suspicious">⚠️ SUSPICIOUS</span>'
    return '<span class="badge badge-safe">✅ AUTHENTIC</span>'


# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>🏛️ Menu</h2>", unsafe_allow_html=True)
    page = st.radio("", [
        "🏠 Dashboard Overview",
        "📝 Intelligence Analysis",
        "📰 Fact Checker",
        "📊 Command Center",
        "📁 Datastore",
        "⚙️ Engine Status"
    ], label_visibility="collapsed")

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔗 Developer Links")
    st.markdown(f"[{'📄 FastAPI Documentation'}]({API_URL}/docs)")
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 12px;">'
        '<p style="color:#94a3b8;font-size:0.85rem;margin:0;"><strong>Smart Governance AI v2.0</strong><br>'
        '<span style="font-size:0.75rem; color:#64748b;">Powered by XLM-RoBERTa + BERTopic</span></p>'
        '</div>',
        unsafe_allow_html=True
    )


# ── Pages ───────────────────────────────────────────────────────────────

if page == "🏠 Dashboard Overview":
    st.markdown('<h1 class="hero-title">Smart Governance AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Next-Generation Civic Intelligence & Misinformation Detection Platform</p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    health = call_api("/health", method="GET")
    stats = call_api("/stats", method="GET")

    api_ok = health is not None
    c1.metric("Engine Status", "🟢 Online" if api_ok else "🔴 Offline")
    if stats:
        c2.metric("Civic Reports Processed", f"{stats.get('complaint_index_size', 0):,}")
        c3.metric("News Articles Indexed", f"{stats.get('news_index_size', 0):,}")
    else:
        c2.metric("Civic Reports Processed", "N/A")
        c3.metric("News Articles Indexed", "N/A")

    models_count = sum(health["models_loaded"].values()) if health else 0
    c4.metric("AI Models Active", f"{models_count}/5")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚡ Live Intelligence Console")
    
    col_input, col_action = st.columns([4, 1])
    with col_input:
        quick_text = st.text_area("", height=120,
                                placeholder="Enter a public grievance, civic report, or news headline for instant AI analysis...", label_visibility="collapsed")
    with col_action:
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        analyze_btn = st.button("Analyze Now", use_container_width=True)

    if analyze_btn:
        if quick_text.strip():
            with st.spinner("AI Engine processing text..."):
                result = call_api("/analyze-text", {"text": quick_text})
            if result:
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""<div class="result-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <h4 style="margin: 0; color: #818cf8;">📂 Categorization</h4>
                        </div>
                        <p class="stat-number" style="font-size:1.8rem; margin: 15px 0;">{result['topic'].get('topic_label','Unknown')}</p>
                        <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                            <span style="color:#94a3b8;">Confidence</span>
                            <span style="color:#10b981; font-weight: bold;">{result['topic'].get('confidence',0):.1%}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    urg = result['urgency'].get('urgency', 'LOW')
                    st.markdown(f"""<div class="result-card urgency-{urg.lower()}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <h4 style="margin: 0; color: #f43f5e;">🚨 Priority Level</h4>
                        </div>
                        <div style="margin: 20px 0;">{get_urgency_badge(urg)}</div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                            <span style="color:#94a3b8;">Confidence</span>
                            <span style="color:#10b981; font-weight: bold;">{result['urgency'].get('confidence',0):.1%}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                with col3:
                    is_sus = result['similarity'].get('is_suspicious', False)
                    card_class = "urgency-critical" if is_sus else "urgency-low"
                    st.markdown(f"""<div class="result-card {card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <h4 style="margin: 0; color: #38bdf8;">🛡️ Integrity Check</h4>
                        </div>
                        <div style="margin: 20px 0;">{get_suspicion_badge(is_sus)}</div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                            <span style="color:#94a3b8;">Matches Found</span>
                            <span style="color:#e2e8f0; font-weight: bold;">{result['similarity'].get('num_similar',0)} sources</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.error("⚠️ AI Engine is disconnected. Please initialize the backend server.")

    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)
    st.markdown("### 🧩 Core Architectures")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""<div class="result-card" style="padding: 25px;">
            <h4 style="color:#c084fc; margin-bottom: 15px;">🏢 Civic Intelligence</h4>
            <div style="color:#cbd5e1; line-height: 1.8;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color:#10b981;">✓</span> Neural Auto-categorization
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color:#10b981;">✓</span> XLM-R Urgency Scaling
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color:#10b981;">✓</span> Native Multilingual Pipeline
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="result-card" style="padding: 25px;">
            <h4 style="color:#38bdf8; margin-bottom: 15px;">📰 Integrity Detector</h4>
            <div style="color:#cbd5e1; line-height: 1.8;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color:#10b981;">✓</span> FAISS Vector Indexing
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color:#10b981;">✓</span> Advanced Paraphrase Matching
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color:#10b981;">✓</span> Cross-lingual Defenses
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="result-card" style="padding: 25px;">
            <h4 style="color:#818cf8; margin-bottom: 15px;">📊 Quantum Analytics</h4>
            <div style="color:#cbd5e1; line-height: 1.8;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color:#10b981;">✓</span> Dynamic Topic Clusters
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color:#10b981;">✓</span> Geolocation Heatmaps
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color:#10b981;">✓</span> Millisecond Telemetry
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


elif page == "📝 Intelligence Analysis":
    st.markdown("<h2 style='font-size: 2.5rem; background: linear-gradient(90deg, #c084fc, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Intelligence Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>Deep-dive semantic analysis of citizen complaints for routing and prioritization.</p>", unsafe_allow_html=True)

    text = st.text_area("Source Text", height=200,
                         placeholder="Enter the full text of the civic report or grievance here...", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    analyze_topic = col1.button("🏷️ Extract Department Routing", use_container_width=True)
    analyze_urgency = col2.button("🚨 Determine SLA Priority", use_container_width=True)

    if analyze_topic and text.strip():
        with st.spinner("Executing neural topic extraction..."):
            result = call_api("/predict-topic", {"text": text})
        if result:
            st.markdown(f"""<div class="result-card">
                <h3 style="color: #818cf8; margin-bottom: 5px;">Primary Department: {result.get('topic_label', 'Unknown')}</h3>
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; margin-top: 15px;">
                    <p style="margin: 5px 0;"><strong>System ID:</strong> {result.get('topic_id', -1)}</p>
                    <p style="margin: 5px 0;"><strong>Confidence Score:</strong> <span style="color: #10b981;">{result.get('confidence', 0):.2%}</span></p>
                    <p style="margin: 5px 0;"><strong>Execution Engine:</strong> {result.get('method', 'model')}</p>
                </div>
            </div>""", unsafe_allow_html=True)

    if analyze_urgency and text.strip():
        with st.spinner("Calculating threat vector & priority..."):
            result = call_api("/predict-urgency", {"text": text})
        if result:
            urg = result.get('urgency', 'LOW')
            st.markdown(f"""<div class="result-card urgency-{urg.lower()}">
                <div style="display: flex; align-items: center; gap: 20px;">
                    <h3 style="margin: 0;">Priority Assignment:</h3>
                    {get_urgency_badge(urg)}
                </div>
                <p style="margin-top: 15px; color: #94a3b8;">Resolution Confidence: <strong style="color: #f8fafc;">{result.get('confidence', 0):.2%}</strong></p>
            </div>""", unsafe_allow_html=True)

            probs = result.get('probabilities', {})
            if probs:
                st.markdown("<br><h4>Probability Distribution Vector</h4>", unsafe_allow_html=True)
                fig = go.Figure(go.Bar(
                    x=list(probs.keys()), y=list(probs.values()),
                    marker_color=['#22c55e', '#eab308', '#f97316', '#ef4444'],
                    text=[f"{v:.1%}" for v in probs.values()], textposition='auto',
                    marker_line_color='rgba(255,255,255,0.2)', marker_line_width=1,
                    opacity=0.8
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0', font_family="Outfit",
                    yaxis_range=[0, 1],
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)


elif page == "📰 Fact Checker":
    st.markdown("<h2 style='font-size: 2.5rem; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Integrity & Misinformation Engine</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>Scan media broadcasts against our FAISS vectorized truth database.</p>", unsafe_allow_html=True)

    news_text = st.text_area("Broadcast Source", height=200,
                              placeholder="Paste the broadcast transcript, article, or headline...", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Initiate Integrity Scan", use_container_width=True):
        if news_text.strip():
            with st.spinner("Querying vector database..."):
                result = call_api("/similar-news", {"text": news_text})
            if result:
                is_sus = result.get('is_suspicious', False)
                main_class = "urgency-critical" if is_sus else "urgency-low"
                
                st.markdown(f"""<div class="result-card {main_class}">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <h3 style="margin: 0;">Scan Verdict</h3>
                        {get_suspicion_badge(is_sus)}
                    </div>
                    <div style="margin-top: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                        <div style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px;">
                            <span style="color:#94a3b8; font-size: 0.9rem;">Confidence Score</span><br>
                            <span style="font-size: 1.2rem; font-weight: bold; color: #f8fafc;">{result.get('confidence', 0):.1%}</span>
                        </div>
                        <div style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px;">
                            <span style="color:#94a3b8; font-size: 0.9rem;">Vector Matches</span><br>
                            <span style="font-size: 1.2rem; font-weight: bold; color: #f8fafc;">{result.get('num_similar', 0)} sources</span>
                        </div>
                    </div>
                    <div style="margin-top: 15px; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px;">
                        <span style="color:#94a3b8; font-size: 0.9rem;">Analysis Reason</span><br>
                        <span style="font-size: 1.1rem; color: #e2e8f0;">{result.get('reason', 'N/A')}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                similar = result.get('similar_articles', [])
                if similar:
                    st.markdown("<br><h3 style='color: #cbd5e1;'>📄 Extracted Source Matches</h3>", unsafe_allow_html=True)
                    for i, article in enumerate(similar):
                        st.markdown(f"""<div class="result-card" style="padding: 20px;">
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; margin-bottom: 10px;">
                                <span style="font-weight: bold; color: #818cf8;">Match #{i+1}</span>
                                <span style="color: #10b981; font-family: monospace;">Distance: {article.get('score', 0):.4f}</span>
                            </div>
                            <p style="color: #e2e8f0; line-height: 1.6; font-size: 0.95rem;">{article.get('text', 'N/A')}</p>
                        </div>""", unsafe_allow_html=True)
            else:
                st.error("⚠️ AI Engine disconnected or Vector Index unreachable.")


elif page == "📊 Command Center":
    st.markdown("<h2 style='font-size: 2.5rem; background: linear-gradient(90deg, #f472b6, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Global Command Center</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>Macro-level intelligence metrics across all monitored jurisdictions.</p>", unsafe_allow_html=True)

    # Sample data for demo visualization
    topic_data = pd.DataFrame({
        'Topic': ['Water Infrastructure', 'Sanitation', 'Road Networks', 'Power Grid', 'Public Transit', 'Noise Pollution', 'Zoning', 'Emergency Svcs'],
        'Incidents': [245, 189, 312, 156, 98, 67, 134, 89],
        'Threat Score': [0.72, 0.45, 0.81, 0.68, 0.35, 0.28, 0.52, 0.61]
    })

    urgency_data = pd.DataFrame({
        'Level': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
        'Volume': [420, 380, 290, 200]
    })
    
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(topic_data, x='Topic', y='Incidents', color='Threat Score',
                     color_continuous_scale='Sunsetdark', title="Incident Volume by Department")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', font_family="Outfit",
                          xaxis=dict(gridcolor='rgba(255,255,255,0.05)'), yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(urgency_data, values='Volume', names='Level',
                     color='Level', color_discrete_map={'LOW':'#10b981','MEDIUM':'#eab308','HIGH':'#f97316','CRITICAL':'#ef4444'},
                     title="Global Threat Distribution", hole=0.55)
        fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#0f172a', width=2)))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', font_family="Outfit",
                          showlegend=False, annotations=[dict(text='Total<br>1290', x=0.5, y=0.5, font_size=20, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    lang_data = pd.DataFrame({'Language Dialect': ['English','Hindi','Telugu','Tamil'], 'Volume': [580, 320, 210, 180]})
    fig = px.bar(lang_data, x='Language Dialect', y='Volume', color='Language Dialect',
                 color_discrete_sequence=['#818cf8','#c084fc','#f472b6','#38bdf8'], title="Multilingual Processing Distribution")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', font_family="Outfit",
                      xaxis=dict(gridcolor='rgba(255,255,255,0.05)'), yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


elif page == "📁 Datastore":
    st.markdown("<h2 style='font-size: 2.5rem; background: linear-gradient(90deg, #e879f9, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Secure Datastore Ingestion</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>Batch ingest external telemetry data directly into the AI pipeline.</p>", unsafe_allow_html=True)

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    data_type = st.selectbox("Pipeline Target", ["civic_reports", "media_broadcasts"])
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload encrypted CSV payload", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"<br><h4 style='color: #818cf8;'>Data Snapshot ({len(df)} records, {len(df.columns)} dimensions)</h4>", unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Initialize Neural Processing", use_container_width=True):
            st.info("Pipeline connected. Awaiting stream validation from FastAPI server.")
    st.markdown("</div>", unsafe_allow_html=True)


elif page == "⚙️ Engine Status":
    st.markdown("<h2 style='font-size: 2.5rem; background: linear-gradient(90deg, #94a3b8, #e2e8f0); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>System Diagnostics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>Real-time telemetry and health diagnostics of the AI microservices.</p>", unsafe_allow_html=True)

    health = call_api("/health", method="GET")
    
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    if health:
        st.markdown("<h3 style='color: #10b981; display: flex; align-items: center; gap: 10px;'><span>🟢</span> Core Engine Online</h3>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        models = health.get('models_loaded', {})
        st.markdown("<h4>Neural Weights Status</h4>", unsafe_allow_html=True)
        
        cols = st.columns(3)
        idx = 0
        for name, loaded in models.items():
            icon = "✅" if loaded else "❌"
            color = "#10b981" if loaded else "#ef4444"
            with cols[idx % 3]:
                st.markdown(f"""
                <div style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 12px; margin-bottom: 15px; border-left: 3px solid {color};">
                    <span style="font-size: 1.2rem;">{icon}</span> 
                    <strong style="color: #e2e8f0; margin-left: 10px;">{name.replace('_', ' ').upper()}</strong>
                </div>
                """, unsafe_allow_html=True)
            idx += 1
    else:
        st.markdown("<h3 style='color: #ef4444; display: flex; align-items: center; gap: 10px;'><span>🔴</span> Core Engine Offline</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8;'>The FastAPI orchestration layer is currently unreachable.</p>", unsafe_allow_html=True)
        st.code("cd smart-governance-ai\npython -m api.main", language="bash")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Infrastructure Playbook")
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    steps = [
        ("Initialize runtime environments", "pip install -r requirements.txt"),
        ("Provision raw datastores", "complaints.csv, fake_news.csv -> data/raw/"),
        ("Execute NLP preprocessors", "python -c \"from src.preprocessing import TextPreprocessor; ...\""),
        ("Generate semantic vectors", "python -c \"from src.embeddings import EmbeddingGenerator; ...\""),
        ("Compile topic architecture", "Execute notebooks/04_topic_modeling.ipynb"),
        ("Fine-tune XLM-R models", "Execute notebooks/05_urgency_training.ipynb"),
        ("Construct FAISS space", "Execute notebooks/06_faiss_similarity.ipynb"),
        ("Boot orchestration API", "python -m api.main"),
        ("Launch visual telemetry", "streamlit run dashboard/app.py"),
    ]
    
    for step, cmd in steps:
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
            <span style="color: #cbd5e1; font-weight: 500;">{step}</span>
            <code style="background: rgba(0,0,0,0.3); color: #818cf8; padding: 5px 10px; border-radius: 6px;">{cmd}</code>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
