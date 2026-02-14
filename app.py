import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Optional: PubChem autocomplete
try:
    import pubchempy as pcp
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False

import os
from dotenv import load_dotenv
load_dotenv()

# --- PAGE CONFIG (must be first Streamlit command) ---
st.set_page_config(page_title="TGR Activity AI", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

## Langsmith Tracking
lc_api_key = os.getenv("LC_API_KEY")
if lc_api_key:
    os.environ['LANGCHAIN_API_KEY'] = lc_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Chat bot for TGR Activity Predictor"

q_api_key = os.getenv("Q_API_KEY")

# Load the trained Random Forest model
model = joblib.load("random_forest_tgr.pkl")

# Function to convert SMILES to fingerprint
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)).reshape(1, -1)
    else:
        return None

## Prompt template for chatbot
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful assistant for the TGR Activity Predictor application. Your role is to help users understand how to use the predictor, explain features, answer questions about TGR (Thioredoxin Glutathione Reductase), guide them through the prediction process, and provide general assistance. You do not make predictions yourself - the prediction model does that.

Additional capabilities:
1. If a user provides a SMILES notation (like c1ccccc1, CCO, CC(=O)O, etc.), identify and tell them the name of the compound.
2. If a user provides the name of a chemical compound (like benzene, ethanol, aspirin, etc.), generate its SMILES notation.

Please respond to user queries in a friendly and helpful manner."""),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, api_key):
    if not api_key:
        return "⚠️ Please configure your Groq API Key in the settings above to use the chatbot."

    try:
        model_chat = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=api_key
        )
        output_parser = StrOutputParser()
        chain = prompt | model_chat | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        return f"❌ Error: {str(e)}. Please check your API key in settings."

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        /* Google Fonts Import */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* General page background and font */
        * {
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        }

        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }

        /* Remove vertical centering - align content to top */
        [data-testid="stAppViewContainer"] > .main {
            min-height: auto !important;
        }

        .main .block-container {
            padding-top: 1rem !important;
            min-height: auto !important;
            display: flex;
            flex-direction: column;
            align-items: stretch;
        }

        [data-testid="stVerticalBlock"] {
            gap: 0.5rem;
        }

        /* Remove extra spacing from main area */
        .stApp > header + div {
            min-height: auto !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e, #16213e);
            border-right: 1px solid rgba(0, 224, 255, 0.1);
            min-width: 380px !important;
            max-width: 380px !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            width: 380px !important;
        }

        [data-testid="collapsedControl"] {
            display: none;
        }

        /* Mobile: show collapse control */
        @media (max-width: 768px) {
            [data-testid="collapsedControl"] {
                display: block !important;
            }

            [data-testid="stSidebar"] {
                min-width: 0 !important;
                max-width: 85% !important;
            }

            [data-testid="stSidebar"] > div:first-child {
                width: 100% !important;
            }
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: #e0e0e0;
        }

        /* Glass Card Effect */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .glass-card-highlight {
            background: linear-gradient(135deg, rgba(0, 224, 255, 0.08) 0%, rgba(0, 119, 255, 0.05) 100%);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid rgba(0, 224, 255, 0.2);
            padding: 2.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 8px 32px rgba(0, 224, 255, 0.1);
        }

        /* Main Title */
        .main-title {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00e0ff 0%, #0077ff 50%, #00e0ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }

        .main-subtitle {
            font-size: 1.15rem;
            color: rgba(255, 255, 255, 0.7);
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 400;
            line-height: 1.6;
        }

        .main-subtitle strong {
            color: #00e0ff;
        }

        /* Section Headers */
        .section-header {
            font-size: 1.3rem;
            font-weight: 600;
            color: #00e0ff;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Info Cards */
        .info-card {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 1.2rem;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
        }

        .info-card:hover {
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(0, 224, 255, 0.2);
            transform: translateY(-2px);
        }

        .info-card-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: #00e0ff;
            margin-bottom: 0.5rem;
        }

        .info-card-text {
            font-size: 0.85rem;
            color: rgba(255, 255, 255, 0.6);
            line-height: 1.5;
        }

        /* SMILES Example Pills */
        .smiles-pill {
            display: inline-block;
            background: rgba(0, 224, 255, 0.1);
            border: 1px solid rgba(0, 224, 255, 0.3);
            border-radius: 20px;
            padding: 0.4rem 1rem;
            margin: 0.25rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #00e0ff;
            transition: all 0.2s ease;
        }

        .smiles-pill:hover {
            background: rgba(0, 224, 255, 0.2);
            transform: scale(1.02);
        }

        /* Input styling */
        .stTextInput > div > div > input {
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.95);
            border: 2px solid rgba(0, 224, 255, 0.3);
            color: #1a1a2e;
            font-family: 'JetBrains Mono', monospace;
            font-size: 1rem;
            padding: 0.8rem 1rem;
            transition: all 0.3s ease;
        }

        .stTextInput > div > div > input:focus {
            border-color: #00e0ff;
            border-width: 2.5px;
            box-shadow: 0 0 20px rgba(0, 224, 255, 0.35);
        }

        /* Predict button */
        .stButton > button {
            background: linear-gradient(135deg, #00e0ff 0%, #0077ff 100%);
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            border-radius: 12px;
            padding: 0.8rem 2.5rem;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 224, 255, 0.2);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #0077ff 0%, #00e0ff 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 224, 255, 0.25);
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        /* Result Cards */
        .result-active {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(16, 185, 129, 0.1) 100%);
            border: 2px solid rgba(34, 197, 94, 0.5);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            animation: pulse-green 2s infinite;
        }

        .result-inactive {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, rgba(180, 30, 30, 0.18) 100%);
            border: 2px solid rgba(239, 68, 68, 0.5);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            animation: pulse-red 2s infinite;
        }

        .result-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .result-active .result-title {
            color: #22c55e;
        }

        .result-inactive .result-title {
            color: #ef4444;
        }

        .result-subtitle {
            font-size: 0.95rem;
            color: rgba(255, 255, 255, 0.7);
        }

        @keyframes pulse-green {
            0%, 100% { box-shadow: 0 0 20px rgba(34, 197, 94, 0.3); }
            50% { box-shadow: 0 0 40px rgba(34, 197, 94, 0.5); }
        }

        @keyframes pulse-red {
            0%, 100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.3); }
            50% { box-shadow: 0 0 40px rgba(239, 68, 68, 0.5); }
        }

        /* Chemical Filling Animation */
        .chemical-fill-container {
            position: relative;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 16px;
            border: 2px solid rgba(0, 224, 255, 0.3);
            padding: 2rem;
            overflow: hidden;
            min-height: 120px;
        }

        .chemical-fill {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 0%;
            background: linear-gradient(180deg,
                rgba(0, 224, 255, 0.1) 0%,
                rgba(0, 224, 255, 0.3) 50%,
                rgba(0, 119, 255, 0.4) 100%);
            animation: fillUp 1.5s ease-out forwards;
            border-radius: 0 0 14px 14px;
        }

        .chemical-fill::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 8px;
            background: linear-gradient(90deg,
                transparent 0%,
                rgba(255, 255, 255, 0.4) 50%,
                transparent 100%);
            animation: shimmer 1s ease-in-out infinite;
        }

        .chemical-fill-active {
            background: linear-gradient(180deg,
                rgba(34, 197, 94, 0.1) 0%,
                rgba(34, 197, 94, 0.3) 50%,
                rgba(16, 185, 129, 0.5) 100%);
        }

        .chemical-fill-inactive {
            background: linear-gradient(180deg,
                rgba(239, 68, 68, 0.1) 0%,
                rgba(239, 68, 68, 0.3) 50%,
                rgba(220, 38, 38, 0.5) 100%);
        }

        @keyframes fillUp {
            0% { height: 0%; }
            100% { height: 100%; }
        }

        @keyframes shimmer {
            0%, 100% { opacity: 0.3; transform: translateX(-100%); }
            50% { opacity: 1; transform: translateX(100%); }
        }

        .bubbles {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            pointer-events: none;
        }

        .bubble {
            position: absolute;
            bottom: -20px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            animation: rise 2s ease-in infinite;
        }

        @keyframes rise {
            0% { bottom: -20px; opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { bottom: 100%; opacity: 0; }
        }

        .result-content {
            position: relative;
            z-index: 10;
            animation: fadeInUp 0.5s ease-out 1s forwards;
            opacity: 0;
        }

        @keyframes fadeInUp {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        .analyzing-text {
            position: relative;
            z-index: 10;
            text-align: center;
            color: #00e0ff;
            font-size: 1.1rem;
            font-weight: 500;
            animation: pulse-text 1s ease-in-out infinite;
        }

        @keyframes pulse-text {
            0%, 100% { opacity: 0.7; }
            50% { opacity: 1; }
        }

        /* Stats Display */
        .stat-box {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 1.5rem;
            text-align: center;
        }

        .stat-number {
            font-size: 2rem;
            font-weight: 700;
            color: #00e0ff;
        }

        .stat-label {
            font-size: 0.85rem;
            color: rgba(255, 255, 255, 0.5);
            margin-top: 0.5rem;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            color: rgba(255, 255, 255, 0.4);
            font-size: 0.85rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            margin-top: 3rem;
        }

        .footer a {
            color: #00e0ff;
            text-decoration: none;
        }

        /* Chat styling */
        .stChatMessage {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
        }

        /* Suggestion buttons styling */
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            background: rgba(0, 224, 255, 0.08) !important;
            border: 1px solid rgba(0, 224, 255, 0.25) !important;
            color: #00e0ff !important;
            font-size: 0.85rem !important;
            padding: 0.4rem 0.8rem !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }

        div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
            background: rgba(0, 224, 255, 0.15) !important;
            border-color: rgba(0, 224, 255, 0.4) !important;
            transform: translateY(-1px);
        }

        /* Divider */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(0, 224, 255, 0.3), transparent);
            margin: 1.5rem 0;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* ============================================================
           LIQUID DOTS OVERLAY
           ============================================================
           Very sparse, randomly distributed liquid dots flowing
           top-right → bottom-left. Uses prime-number spacing for
           pseudo-random appearance.

           TWEAKABLE:
           - Dot count: Increase background-size for fewer dots
           - Randomness: Adjust prime-based spacing values
           - Flow speed: Change animation duration
           ============================================================ */

        /* Main overlay container */
        .laminar-flow-overlay {
            position: fixed;
            inset: 0;
            z-index: 1;
            pointer-events: none;
            overflow: hidden;
            background: transparent;
        }

        /* Layer 1: Sparse random dots */
        .laminar-flow-overlay::before {
            content: '';
            position: absolute;
            top: -100%;
            left: -100%;
            width: 300%;
            height: 300%;
            background-image:
                radial-gradient(circle 4px at center, rgba(0, 230, 255, 0.45) 0%, rgba(0, 220, 255, 0.2) 40%, transparent 70%),
                radial-gradient(circle 14px at center, rgba(0, 210, 245, 0.12) 0%, transparent 75%);
            background-size: 397px 433px, 397px 433px;
            background-position: 0 0, 0 0;
            animation: liquidFlow 50s linear infinite;
            will-change: transform;
        }

        /* Layer 2: Different prime spacing - creates randomness */
        .laminar-flow-overlay::after {
            content: '';
            position: absolute;
            top: -100%;
            left: -100%;
            width: 300%;
            height: 300%;
            background-image:
                radial-gradient(circle 3px at center, rgba(0, 220, 255, 0.4) 0%, rgba(0, 210, 250, 0.15) 50%, transparent 75%),
                radial-gradient(circle 11px at center, rgba(0, 215, 250, 0.1) 0%, transparent 80%);
            background-size: 509px 547px, 509px 547px;
            background-position: 173px 211px, 173px 211px;
            animation: liquidFlow 65s linear infinite;
            animation-delay: -25s;
            will-change: transform;
        }

        /* Liquid flow animation */
        @keyframes liquidFlow {
            0% {
                transform: translateX(33.33%) translateY(-33.33%);
            }
            100% {
                transform: translateX(-33.33%) translateY(33.33%);
            }
        }

        /* Layer 3: Very sparse accent */
        .flow-layer-2 {
            position: fixed;
            top: -100%;
            left: -100%;
            width: 300%;
            height: 300%;
            z-index: 1;
            pointer-events: none;
            background-image:
                radial-gradient(circle 5px at center, rgba(0, 235, 255, 0.35) 0%, rgba(0, 225, 255, 0.12) 50%, transparent 75%),
                radial-gradient(circle 16px at center, rgba(0, 220, 250, 0.07) 0%, transparent 70%);
            background-size: 631px 673px, 631px 673px;
            background-position: 89px 127px, 89px 127px;
            animation: liquidFlow 80s linear infinite;
            animation-delay: -40s;
            will-change: transform;
        }

        /* Layer 4: Extra sparse large drops */
        .flow-layer-3 {
            position: fixed;
            top: -100%;
            left: -100%;
            width: 300%;
            height: 300%;
            z-index: 1;
            pointer-events: none;
            background-image:
                radial-gradient(circle 6px at center, rgba(0, 230, 255, 0.3) 0%, rgba(0, 220, 255, 0.1) 45%, transparent 70%),
                radial-gradient(circle 20px at center, rgba(0, 215, 250, 0.05) 0%, transparent 65%);
            background-size: 743px 787px, 743px 787px;
            background-position: 317px 53px, 317px 53px;
            animation: liquidFlow 95s linear infinite;
            animation-delay: -15s;
            will-change: transform;
        }

        /* Subtle ambient glow */
        .laminar-glow {
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            background:
                radial-gradient(
                    ellipse 45% 22% at 72% 28%,
                    rgba(0, 200, 230, 0.01) 0%,
                    transparent 60%
                ),
                radial-gradient(
                    ellipse 35% 18% at 22% 72%,
                    rgba(0, 210, 235, 0.008) 0%,
                    transparent 55%
                );
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.02);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(0, 224, 255, 0.3);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 224, 255, 0.5);
        }

        /* ============================================================
           MAGIC LINE NAVIGATION - Top bar, no outer container
           ============================================================ */
        .magic-line-nav {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 0.75rem 2rem;
            margin: 0;
            width: 100%;
            position: relative;
            background: transparent;
        }

        .magic-line-nav-inner {
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            position: relative;
        }

        .magic-line-nav a {
            position: relative;
            padding: 0.75rem 1.5rem;
            color: rgba(255, 255, 255, 0.6);
            text-decoration: none;
            font-size: 1rem;
            font-weight: 500;
            letter-spacing: 0.02em;
            transition: color 0.3s ease;
            z-index: 2;
            cursor: pointer;
            border-radius: 8px;
        }

        .magic-line-nav a:hover {
            color: rgba(255, 255, 255, 0.9);
        }

        .magic-line-nav a.active {
            color: #ffffff;
        }

        /* Hide the line - only keep pill */
        .magic-line {
            display: none;
        }

        .magic-line-pill {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            height: calc(100% - 4px);
            background: linear-gradient(135deg, rgba(0, 224, 255, 0.12), rgba(0, 119, 255, 0.08));
            border: 1px solid rgba(0, 224, 255, 0.3);
            border-radius: 10px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 1;
        }

        /* Mobile responsive */
        @media (max-width: 768px) {
            .magic-line-nav {
                padding: 0.5rem 1rem;
            }
            .magic-line-nav a {
                padding: 0.5rem 1rem;
                font-size: 0.9rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Auto-collapse sidebar on mobile
components.html(
    """
    <script>
        (function() {
            // ============================================================
            // MICROFLUIDIC CHANNEL FLOW - Inject overlay layers
            // ============================================================

            // Layer 1: Ambient glow (static)
            if (!window.parent.document.getElementById('laminarGlow')) {
                const glow = document.createElement('div');
                glow.id = 'laminarGlow';
                glow.className = 'laminar-glow';
                window.parent.document.body.insertBefore(glow, window.parent.document.body.firstChild);
            }

            // Layer 2: Main flow overlay (channels + primary flow)
            if (!window.parent.document.getElementById('laminarFlowOverlay')) {
                const overlay = document.createElement('div');
                overlay.id = 'laminarFlowOverlay';
                overlay.className = 'laminar-flow-overlay';
                window.parent.document.body.insertBefore(overlay, window.parent.document.body.firstChild.nextSibling);
            }

            // Layer 3: Secondary flow channel (depth layer)
            if (!window.parent.document.getElementById('flowLayer2')) {
                const flow2 = document.createElement('div');
                flow2.id = 'flowLayer2';
                flow2.className = 'flow-layer-2';
                window.parent.document.body.insertBefore(flow2, window.parent.document.body.firstChild.nextSibling);
            }

            // Layer 4: Tertiary flow channel (more depth)
            if (!window.parent.document.getElementById('flowLayer3')) {
                const flow3 = document.createElement('div');
                flow3.id = 'flowLayer3';
                flow3.className = 'flow-layer-3';
                window.parent.document.body.insertBefore(flow3, window.parent.document.body.firstChild.nextSibling);
            }

            // Mobile sidebar auto-collapse
            const isMobile = window.innerWidth <= 768;
            if (isMobile) {
                // Wait for Streamlit to fully render
                setTimeout(function() {
                    const collapseBtn = window.parent.document.querySelector('[data-testid="collapsedControl"] button');
                    const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');

                    // Check if sidebar is currently expanded (has width)
                    if (sidebar && collapseBtn) {
                        const sidebarWidth = sidebar.getBoundingClientRect().width;
                        if (sidebarWidth > 50) {
                            // Sidebar is open, click to close it
                            collapseBtn.click();
                        }
                    }
                }, 100);
            }
        })();
    </script>
    """,
    height=0
)

# --- PAGE STATE MANAGEMENT ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "predictor"

# Define pages
PAGES = {
    "predictor": {"title": "🔬 Predictor", "icon": "🔬"},
    "analysis": {"title": "📊 Analysis", "icon": "📊"},
    "about": {"title": "ℹ️ About", "icon": "ℹ️"},
}

def set_page(page_name):
    st.session_state.current_page = page_name

# Navigation will be rendered after hero header below

# Hide the Streamlit nav buttons CSS
st.markdown("""
<style>
    /* Hide the navigation buttons */
    div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) {
        position: absolute;
        opacity: 0;
        pointer-events: none;
        height: 0;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CHATBOT ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 1.5rem; font-weight: 600; color: #00e0ff;">💬 AI Assistant</div>
        <div style="font-size: 0.85rem; color: rgba(255,255,255,0.5); margin-top: 0.25rem;">Powered by Groq LLM</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = None

    # API Key Configuration Section
    with st.expander("🔑 API Key Settings", expanded=not st.session_state.groq_api_key):
        st.markdown("**Groq API Key Configuration**")
        st.caption("The chatbot requires your Groq API key to function.")

        if st.session_state.groq_api_key:
            st.success("✓ API Key configured - Chatbot is ready!")
        else:
            st.warning("⚠ No API Key set - Chatbot is disabled")

        api_key_input = st.text_input(
            "Enter your Groq API Key:",
            value="",
            type="password",
            placeholder="gsk_...",
            help="Get your free API key from https://console.groq.com/keys"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Key", use_container_width=True):
                if api_key_input:
                    st.session_state.groq_api_key = api_key_input
                    st.success("API Key saved!")
                    st.rerun()
                else:
                    st.error("Please enter a valid API key")

        with col2:
            if st.button("🗑️ Clear Key", use_container_width=True):
                st.session_state.groq_api_key = None
                st.session_state.messages = []
                st.success("API Key cleared")
                st.rerun()

    with st.expander("ℹ️ About Assistant"):
        st.info("""
        **I can help you with:**
        - How to use the predictor
        - Understanding TGR
        - Convert SMILES ↔ Names
        - Interpreting results

        **Examples:**
        - "What is c1ccccc1?"
        - "SMILES for aspirin?"
        - "How does TGR work?"
        """)

    st.markdown("---")

    if not st.session_state.groq_api_key:
        st.info("💡 **Chatbot is disabled**\n\nPlease add your Groq API key in the settings above to start chatting.")
    else:
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        question = st.chat_input("Ask me anything...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("Thinking..."):
                response = generate_response(question, st.session_state.groq_api_key)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        st.markdown("---")

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()


# --- MAIN UI ---

# =====================================================================
# HERO HEADER (Common for all pages)
# =====================================================================
st.markdown("""
<div class="glass-card" style="text-align: center; padding: 1.5rem 1rem; margin: 0.5rem auto 1rem auto; max-width: 700px;">
    <div class="main-title" style="font-size: 2.5rem; margin-bottom: 0.3rem;">🧪 TGR Activity Predictor</div>
    <div class="main-subtitle" style="font-size: 1rem; margin-bottom: 0;">
        Predict compound activity against <strong>Thioredoxin Glutathione Reductase (TGR)</strong><br>
        using machine learning powered by Morgan Fingerprints
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# NAVIGATION TABS (Between hero and content)
# =====================================================================
current_page = st.session_state.current_page

# Create navigation with magic line effect
nav_html = f"""
<div class="magic-line-nav" id="magicLineNav">
    <div class="magic-line-nav-inner">
        <a data-page="predictor" class="{'active' if current_page == 'predictor' else ''}" onclick="setPage('predictor')">🔬 Predictor</a>
        <a data-page="analysis" class="{'active' if current_page == 'analysis' else ''}" onclick="setPage('analysis')">📊 Analysis</a>
        <a data-page="about" class="{'active' if current_page == 'about' else ''}" onclick="setPage('about')">ℹ️ About</a>
        <div class="magic-line-pill" id="magicLinePill"></div>
        <div class="magic-line" id="magicLine"></div>
    </div>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)

# Magic line JavaScript
components.html(f"""
<script>
(function() {{
    const currentPage = '{current_page}';

    function initMagicLine() {{
        const nav = window.parent.document.getElementById('magicLineNav');
        if (!nav) {{
            setTimeout(initMagicLine, 100);
            return;
        }}

        const links = nav.querySelectorAll('a[data-page]');
        const magicLine = nav.querySelector('#magicLine');
        const magicPill = nav.querySelector('#magicLinePill');
        const navInner = nav.querySelector('.magic-line-nav-inner');

        if (!links.length || !magicLine || !magicPill) return;

        function updateMagicLine(el) {{
            const rect = el.getBoundingClientRect();
            const navRect = navInner.getBoundingClientRect();
            const left = rect.left - navRect.left;
            const width = rect.width;

            magicLine.style.left = left + 'px';
            magicLine.style.width = width + 'px';

            magicPill.style.left = left + 'px';
            magicPill.style.width = width + 'px';
        }}

        // Set initial position
        const activeLink = nav.querySelector('a.active') || links[0];
        updateMagicLine(activeLink);

        // Add hover effects
        links.forEach(link => {{
            link.addEventListener('mouseenter', () => updateMagicLine(link));
            link.addEventListener('mouseleave', () => {{
                const active = nav.querySelector('a.active') || links[0];
                updateMagicLine(active);
            }});
        }});

        // Update on window resize
        window.addEventListener('resize', () => {{
            const active = nav.querySelector('a.active') || links[0];
            updateMagicLine(active);
        }});
    }}

    // Initialize after DOM is ready
    setTimeout(initMagicLine, 100);
}})();

// Handle nav clicks
function setPage(page) {{
    const buttons = window.parent.document.querySelectorAll('button[kind="secondary"]');
    buttons.forEach(btn => {{
        if (btn.innerText.toLowerCase().includes(page)) {{
            btn.click();
        }}
    }});
}}
window.parent.setPage = setPage;

setTimeout(() => {{
    const nav = window.parent.document.getElementById('magicLineNav');
    if (nav) {{
        const links = nav.querySelectorAll('a[data-page]');
        links.forEach(link => {{
            link.onclick = (e) => {{
                e.preventDefault();
                const page = link.getAttribute('data-page');
                setPage(page);
            }};
        }});
    }}
}}, 200);
</script>
""", height=0)

# Navigation buttons (invisible but functional)
nav_cols = st.columns(3)
with nav_cols[0]:
    if st.button("Predictor", key="nav_predictor", use_container_width=True, type="secondary"):
        set_page("predictor")
        st.rerun()
with nav_cols[1]:
    if st.button("Analysis", key="nav_analysis", use_container_width=True, type="secondary"):
        set_page("analysis")
        st.rerun()
with nav_cols[2]:
    if st.button("About", key="nav_about", use_container_width=True, type="secondary"):
        set_page("about")
        st.rerun()

# =====================================================================
# PAGE: PREDICTOR
# =====================================================================
if st.session_state.current_page == "predictor":
    # Initialize session state
    if "selected_smiles" not in st.session_state:
        st.session_state.selected_smiles = ""
    if "pubchem_suggestions" not in st.session_state:
        st.session_state.pubchem_suggestions = []

    # Centered container for Molecular Prediction UI
    _, pred_container, _ = st.columns([1, 2, 1])
    with pred_container:
        # Main Prediction Card
        st.markdown('<div class="section-header" style="text-align: center;">🔬 Molecular Prediction</div>', unsafe_allow_html=True)

        # ========== SMILES INPUT (Main - Outside) ==========
        user_input = st.text_input(
            "Enter SMILES Notation",
            value=st.session_state.selected_smiles,
            placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O",
            help="SMILES (Simplified Molecular Input Line Entry System) is a notation for describing molecular structures"
        )

        # Example SMILES section
        st.markdown("""
        <div style="margin: 1rem 0; text-align: center;">
            <span style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">Try examples:</span>
            <span class="smiles-pill">c1ccccc1</span>
            <span class="smiles-pill">CCO</span>
            <span class="smiles-pill">CC(=O)O</span>
            <span class="smiles-pill">CC(=O)Nc1ccc(O)cc1</span>
        </div>
        """, unsafe_allow_html=True)

        # ========== COMPOUND NAME SEARCH (Inside Expander) ==========
        with st.expander("🔎 Search by Compound Name (PubChem)", expanded=False):
            # Compound name input
            query = st.text_input(
                "Enter Compound Name",
                placeholder="aspirin, caffeine, ibuprofen...",
                key="compound_query"
            )

            # Search button
            search_btn = st.button("🔍 Search PubChem", key="search_pubchem_btn")

            # Show suggestions when search is clicked or query has 2+ chars
            if search_btn and query:
                try:
                    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/{query}/json?limit=8"
                    resp = requests.get(url, timeout=3)
                    if resp.status_code == 200:
                        suggestions = resp.json().get("dictionary_terms", {}).get("compound", [])[:8]
                        if suggestions:
                            st.session_state.pubchem_suggestions = suggestions
                        else:
                            st.warning("No compounds found. Try different spelling.")
                except Exception as e:
                    st.error(f"Search failed: {e}")

            # Show dropdown if suggestions exist
            if "pubchem_suggestions" in st.session_state and st.session_state.pubchem_suggestions:
                selected = st.selectbox(
                    "💡 Select compound:",
                    options=st.session_state.pubchem_suggestions
                )

                if st.button("🔄 Get SMILES", type="primary"):
                    if PUBCHEM_AVAILABLE:
                        try:
                            compounds = pcp.get_compounds(selected, 'name')
                            if compounds:
                                st.session_state.selected_smiles = compounds[0].isomeric_smiles
                                st.session_state.pubchem_suggestions = []  # Clear suggestions
                                st.success(f"✅ {selected} → `{st.session_state.selected_smiles}`")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.warning("Install: pip install pubchempy")

        # Predict button - Centered
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔍 PREDICT ACTIVITY", use_container_width=True, type="primary")

    # Prediction Result
    if predict_button:
        if not user_input or not user_input.strip():
            st.warning("Please enter a SMILES notation to make a prediction.")
        else:
            fp = smiles_to_fp(user_input)
            if fp is None:
                st.error("❌ Invalid SMILES notation. Please check your input and try again.")
            else:
                prediction = model.predict(fp)[0]

                st.markdown("<br>", unsafe_allow_html=True)

                result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
                with result_col2:
                    if prediction == 1:
                        st.markdown("""
                        <div id="prediction-result" class="chemical-fill-container" style="border-color: rgba(34, 197, 94, 0.5);">
                            <div class="chemical-fill chemical-fill-active"></div>
                            <div class="bubbles">
                                <div class="bubble" style="left: 5%; width: 6px; height: 6px; animation-delay: 0s; animation-duration: 2.5s;"></div>
                                <div class="bubble" style="left: 12%; width: 8px; height: 8px; animation-delay: 0.4s; animation-duration: 2.2s;"></div>
                                <div class="bubble" style="left: 22%; width: 5px; height: 5px; animation-delay: 1.2s; animation-duration: 2.8s;"></div>
                                <div class="bubble" style="left: 30%; width: 10px; height: 10px; animation-delay: 0.8s; animation-duration: 2s;"></div>
                                <div class="bubble" style="left: 40%; width: 7px; height: 7px; animation-delay: 1.5s; animation-duration: 2.3s;"></div>
                                <div class="bubble" style="left: 50%; width: 6px; height: 6px; animation-delay: 0.2s; animation-duration: 2.6s;"></div>
                                <div class="bubble" style="left: 58%; width: 9px; height: 9px; animation-delay: 1s; animation-duration: 2.1s;"></div>
                                <div class="bubble" style="left: 68%; width: 5px; height: 5px; animation-delay: 0.6s; animation-duration: 2.7s;"></div>
                                <div class="bubble" style="left: 78%; width: 8px; height: 8px; animation-delay: 1.3s; animation-duration: 2.4s;"></div>
                                <div class="bubble" style="left: 88%; width: 6px; height: 6px; animation-delay: 0.3s; animation-duration: 2.2s;"></div>
                            </div>
                            <div class="result-content">
                                <div class="result-title" style="color: #22c55e; font-size: 1.8rem;">✅ ACTIVE</div>
                                <div class="result-subtitle">This compound shows predicted activity against TGR</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div id="prediction-result" class="chemical-fill-container" style="border-color: rgba(239, 68, 68, 0.5);">
                            <div class="chemical-fill chemical-fill-inactive"></div>
                            <div class="bubbles">
                                <div class="bubble" style="left: 8%; width: 7px; height: 7px; animation-delay: 0.5s; animation-duration: 2.3s;"></div>
                                <div class="bubble" style="left: 18%; width: 5px; height: 5px; animation-delay: 1.1s; animation-duration: 2.6s;"></div>
                                <div class="bubble" style="left: 25%; width: 9px; height: 9px; animation-delay: 0.2s; animation-duration: 2.1s;"></div>
                                <div class="bubble" style="left: 35%; width: 6px; height: 6px; animation-delay: 0.9s; animation-duration: 2.5s;"></div>
                                <div class="bubble" style="left: 45%; width: 8px; height: 8px; animation-delay: 1.4s; animation-duration: 2.2s;"></div>
                                <div class="bubble" style="left: 55%; width: 5px; height: 5px; animation-delay: 0.3s; animation-duration: 2.7s;"></div>
                                <div class="bubble" style="left: 65%; width: 7px; height: 7px; animation-delay: 0.7s; animation-duration: 2.4s;"></div>
                                <div class="bubble" style="left: 72%; width: 10px; height: 10px; animation-delay: 1.2s; animation-duration: 2s;"></div>
                                <div class="bubble" style="left: 82%; width: 6px; height: 6px; animation-delay: 0.1s; animation-duration: 2.8s;"></div>
                                <div class="bubble" style="left: 92%; width: 8px; height: 8px; animation-delay: 0.8s; animation-duration: 2.3s;"></div>
                            </div>
                            <div class="result-content">
                                <div class="result-title" style="color: #ef4444; font-size: 1.8rem;">❌ INACTIVE</div>
                                <div class="result-subtitle">This compound shows no predicted activity against TGR</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Display analyzed compound
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <span style="color: rgba(255,255,255,0.5);">Analyzed Compound:</span>
                    <code style="display: block; margin-top: 0.5rem; font-size: 1.1rem; color: #00e0ff; font-family: 'JetBrains Mono', monospace;">{user_input}</code>
                </div>
                """, unsafe_allow_html=True)

                # Scroll to result
                components.html(
                    """
                    <script>
                        setTimeout(function() {
                            const element = window.parent.document.getElementById('prediction-result');
                            if (element) {
                                element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            }
                        }, 150);
                    </script>
                    """,
                    height=0
                )

    # Info Section
    st.markdown("<br>", unsafe_allow_html=True)

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.markdown("""
        <div class="glass-card">
            <div class="section-header">🎯 About TGR</div>
            <div class="info-card">
                <div class="info-card-title">What is TGR?</div>
                <div class="info-card-text">
                    Thioredoxin Glutathione Reductase is a key enzyme in parasitic organisms,
                    making it an important drug target for treating parasitic diseases.
                </div>
            </div>
            <div class="info-card">
                <div class="info-card-title">Why Target TGR?</div>
                <div class="info-card-text">
                    TGR is essential for parasite survival but absent in humans,
                    making it an ideal selective drug target.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with info_col2:
        st.markdown("""
        <div class="glass-card">
            <div class="section-header">📖 How to Use</div>
            <div class="info-card">
                <div class="info-card-title">Step 1: Get SMILES</div>
                <div class="info-card-text">
                    Obtain the SMILES notation of your compound from databases like PubChem,
                    ChEMBL, or use the AI assistant.
                </div>
            </div>
            <div class="info-card">
                <div class="info-card-title">Step 2: Predict</div>
                <div class="info-card-text">
                    Enter the SMILES string and click Predict to get instant
                    activity prediction results.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with info_col3:
        st.markdown("""
        <div class="glass-card">
            <div class="section-header">⚙️ Model Info</div>
            <div class="info-card">
                <div class="info-card-title">Algorithm</div>
                <div class="info-card-text">
                    Random Forest classifier trained on curated TGR bioactivity data
                    with Morgan fingerprints (2048 bits).
                </div>
            </div>
            <div class="info-card">
                <div class="info-card-title">Features</div>
                <div class="info-card-text">
                    Uses circular fingerprints with radius 2 to capture
                    molecular substructure patterns.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# PAGE: ANALYSIS
# =====================================================================
elif st.session_state.current_page == "analysis":
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 1.5rem 1rem; margin: 0.5rem auto 1.5rem auto; max-width: 700px;">
        <div class="main-title" style="font-size: 2.5rem; margin-bottom: 0.3rem;">📊 Analysis</div>
        <div class="main-subtitle" style="font-size: 1rem; margin-bottom: 0;">
            Batch analysis and detailed compound insights
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="section-header">📈 Batch Prediction</div>
        <p style="color: rgba(255,255,255,0.7);">
            Upload a CSV file with SMILES notations to predict activity for multiple compounds at once.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV with SMILES column", type=['csv'])

    if uploaded_file is not None:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        smiles_col = st.selectbox("Select SMILES column:", df.columns.tolist())

        if st.button("🔬 Run Batch Prediction"):
            with st.spinner("Processing compounds..."):
                results = []
                for smiles in df[smiles_col]:
                    fp = smiles_to_fp(smiles)
                    if fp is not None:
                        pred = model.predict(fp)[0]
                        results.append("Active" if pred == 1 else "Inactive")
                    else:
                        results.append("Invalid SMILES")

                df['Prediction'] = results
                st.success("Batch prediction complete!")
                st.dataframe(df)

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "tgr_predictions.csv",
                    "text/csv"
                )

# =====================================================================
# PAGE: ABOUT
# =====================================================================
elif st.session_state.current_page == "about":
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 1.5rem 1rem; margin: 0.5rem auto 1.5rem auto; max-width: 700px;">
        <div class="main-title" style="font-size: 2.5rem; margin-bottom: 0.3rem;">ℹ️ About</div>
        <div class="main-subtitle" style="font-size: 1rem; margin-bottom: 0;">
            Learn more about the TGR Activity Predictor
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="section-header">🧬 What is TGR?</div>
            <div class="info-card-text" style="color: rgba(255,255,255,0.8); line-height: 1.8;">
                <p><strong style="color: #00e0ff;">Thioredoxin Glutathione Reductase (TGR)</strong> is a unique selenoenzyme
                found in parasitic flatworms such as <em>Schistosoma mansoni</em> and <em>Echinococcus granulosus</em>.</p>

                <p>TGR plays a crucial role in:</p>
                <ul>
                    <li>Redox homeostasis in parasites</li>
                    <li>Defense against oxidative stress</li>
                    <li>Parasite survival and reproduction</li>
                </ul>

                <p>Since TGR is absent in mammals and essential for parasite survival,
                it represents an ideal drug target for developing new antiparasitic therapies.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="section-header">🤖 About the Model</div>
            <div class="info-card-text" style="color: rgba(255,255,255,0.8); line-height: 1.8;">
                <p><strong style="color: #00e0ff;">Machine Learning Approach:</strong></p>
                <ul>
                    <li><strong>Algorithm:</strong> Random Forest Classifier</li>
                    <li><strong>Features:</strong> Morgan Fingerprints (2048 bits, radius 2)</li>
                    <li><strong>Training Data:</strong> Curated TGR bioactivity data</li>
                </ul>

                <p><strong style="color: #00e0ff;">How it works:</strong></p>
                <ol>
                    <li>SMILES notation is converted to molecular structure</li>
                    <li>Morgan fingerprints capture molecular features</li>
                    <li>Random Forest predicts activity/inactivity</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="margin-top: 1rem;">
        <div class="section-header">📚 References & Resources</div>
        <div class="info-card-text" style="color: rgba(255,255,255,0.8);">
            <ul>
                <li><a href="https://pubchem.ncbi.nlm.nih.gov/" target="_blank" style="color: #00e0ff;">PubChem</a> - Find SMILES notations for compounds</li>
                <li><a href="https://www.ebi.ac.uk/chembl/" target="_blank" style="color: #00e0ff;">ChEMBL</a> - Bioactivity database</li>
                <li><a href="https://rdkit.org/" target="_blank" style="color: #00e0ff;">RDKit</a> - Chemistry toolkit used for fingerprint generation</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
