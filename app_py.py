import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import time

# Load the trained Random Forest model
model = joblib.load("random_forest_tgr.pkl")

# Function to convert SMILES to fingerprint
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)).reshape(1, -1)
    else:
        return None


# --- PAGE CONFIG ---
st.set_page_config(page_title="TGR Activity AI", page_icon="🧪", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        /* General page background */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        }

        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }

        /* Hide default padding */
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        /* Style all text elements */
        .stMarkdown, .stMarkdown p, .stMarkdown div {
            color: #ffffff !important;
        }

        /* Input box styling */
        .stTextInput > div > div > input {
            border-radius: 10px !important;
            background: rgba(255,255,255,0.95) !important;
            color: #000000 !important;
            border: 2px solid rgba(0, 224, 255, 0.3) !important;
            padding: 12px !important;
            font-size: 16px !important;
        }

        .stTextInput > div > div > input:focus {
            border-color: #00e0ff !important;
            box-shadow: 0 0 10px rgba(0, 224, 255, 0.5) !important;
        }

        /* Input label */
        .stTextInput > label {
            color: #ffffff !important;
            font-size: 18px !important;
            font-weight: 500 !important;
        }

        /* Predict button */
        .stButton > button {
            background: linear-gradient(90deg, #00e0ff, #0077ff) !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 12px !important;
            padding: 12px 28px !important;
            border: none !important;
            transition: all 0.3s ease !important;
            font-size: 16px !important;
            width: 100%;
            margin-top: 10px;
        }

        .stButton > button:hover {
            background: linear-gradient(90deg, #0077ff, #00e0ff) !important;
            transform: scale(1.02) !important;
            box-shadow: 0 4px 15px rgba(0, 224, 255, 0.4) !important;
        }

        /* Glass container with fill and break animation */
        .glass-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.4);
            text-align: center;
            backdrop-filter: blur(12px);
            max-width: 600px;
            margin: 0 auto;
            position: relative;
            overflow: hidden;
        }

        /* Chemical fill effect */
        @keyframes fillGlass {
            0% { height: 0%; opacity: 0.7; }
            100% { height: 100%; opacity: 0.9; }
        }

        .chemical-fill {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(180deg, rgba(0, 224, 255, 0.3), rgba(0, 119, 255, 0.5));
            animation: fillGlass 2s ease-out forwards;
            z-index: 1;
        }

        /* Glass break effect */
        @keyframes shatter {
            0% { 
                transform: scale(1);
                opacity: 1;
            }
            50% {
                transform: scale(1.05);
            }
            100% { 
                transform: scale(1.2) rotate(5deg);
                opacity: 0;
            }
        }

        @keyframes crackLines {
            0% { opacity: 0; }
            50% { opacity: 1; }
            100% { opacity: 0; }
        }

        .glass-breaking {
            animation: shatter 0.8s ease-out forwards;
        }

        .glass-breaking::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                linear-gradient(45deg, transparent 48%, rgba(255,255,255,0.8) 49%, rgba(255,255,255,0.8) 51%, transparent 52%),
                linear-gradient(-45deg, transparent 48%, rgba(255,255,255,0.8) 49%, rgba(255,255,255,0.8) 51%, transparent 52%),
                linear-gradient(135deg, transparent 48%, rgba(255,255,255,0.6) 49%, rgba(255,255,255,0.6) 51%, transparent 52%);
            animation: crackLines 0.6s ease-out;
            z-index: 10;
            pointer-events: none;
        }

        /* Result reveal animation */
        @keyframes resultReveal {
            0% { 
                transform: scale(0.5) translateY(20px);
                opacity: 0;
            }
            60% {
                transform: scale(1.1) translateY(0);
            }
            100% { 
                transform: scale(1) translateY(0);
                opacity: 1;
            }
        }

        .result-reveal {
            animation: resultReveal 0.6s ease-out forwards;
            animation-delay: 0.4s;
            opacity: 0;
        }

        /* Success/Error messages */
        .stSuccess, .stError {
            background: rgba(255, 255, 255, 0.15) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 15px !important;
            padding: 20px !important;
            margin-top: 20px !important;
            border: 2px solid rgba(255, 255, 255, 0.2) !important;
        }

        .stSuccess p, .stError p {
            color: #ffffff !important;
            font-size: 22px !important;
            font-weight: 600 !important;
        }

        /* Processing state */
        .processing-text {
            color: #00e0ff;
            font-size: 18px;
            margin-top: 20px;
            animation: pulse 1.5s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }

        /* Chatbot floating icon */
        .chat-icon {
            position: fixed;
            bottom: 25px;
            right: 25px;
            background: linear-gradient(135deg, #0077ff, #00e0ff);
            border-radius: 50%;
            width: 60px;
            height: 60px;
            color: white;
            font-size: 28px;
            text-align: center;
            line-height: 60px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            cursor: pointer;
            z-index: 999;
            transition: transform 0.3s ease;
        }

        .chat-icon:hover {
            transform: scale(1.1);
        }

        /* Ensure content stays above fill */
        .content-layer {
            position: relative;
            z-index: 2;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
if 'result_text' not in st.session_state:
    st.session_state.result_text = ""

# --- MAIN UI ---
# Determine glass container classes
glass_classes = "glass-container"
if st.session_state.processing:
    glass_classes = "glass-container"
elif st.session_state.show_result:
    glass_classes = "glass-container glass-breaking"

# Create container
st.markdown(f"""
    <div class='{glass_classes}' id='glassContainer'>
        {'<div class="chemical-fill"></div>' if st.session_state.processing else ''}
        <div class='content-layer'>
            <div style='
                font-size: 2.2em;
                font-weight: 600;
                color: #00e0ff;
                margin-bottom: 10px;
            '>🧪 TGR Activity Predictor</div>
            <div style='
                font-size: 1.1em;
                color: #cceeff;
                margin-bottom: 30px;
            '>Predict whether a compound is <b>Active</b> or <b>Inactive</b> against Thioredoxin Glutathione Reductase (TGR).</div>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Input and prediction section
user_input = st.text_input("👉 Enter SMILES:", "", key="smiles_input")

if st.button("Predict"):
    if not user_input.strip():
        st.error("⚠️ Please enter a SMILES string.")
        st.session_state.processing = False
        st.session_state.show_result = False
    else:
        # Start processing - trigger fill animation
        st.session_state.processing = True
        st.session_state.show_result = False
        st.rerun()

# Handle processing state
if st.session_state.processing:
    # Show processing message
    st.markdown('<div class="processing-text">⚗️ Analyzing molecular structure...</div>', unsafe_allow_html=True)
    
    # Simulate processing time for animation
    time.sleep(2.5)
    
    # Perform prediction
    fp = smiles_to_fp(user_input)
    if fp is None:
        st.session_state.result_text = "error"
        st.session_state.processing = False
        st.session_state.show_result = True
        st.rerun()
    else:
        prediction = model.predict(fp)[0]
        activity = "🟢 Active" if prediction == 1 else "🔴 Inactive"
        st.session_state.result_text = activity
        st.session_state.processing = False
        st.session_state.show_result = True
        st.rerun()

# Show result after glass breaks
if st.session_state.show_result:
    time.sleep(0.5)  # Wait for break animation
    st.markdown('<div class="result-reveal">', unsafe_allow_html=True)
    if st.session_state.result_text == "error":
        st.error("❌ Invalid SMILES string. Please try again.")
    else:
        st.success(f"Prediction: **{st.session_state.result_text}**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Reset state after showing result
    if st.button("Predict Another"):
        st.session_state.processing = False
        st.session_state.show_result = False
        st.session_state.result_text = ""
        st.rerun()

# --- Floating Chatbot Icon ---
st.markdown("""
    <div class="chat-icon" title="Open Chat">
        💬
    </div>
""", unsafe_allow_html=True)
