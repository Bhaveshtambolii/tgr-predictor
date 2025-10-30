import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import time

# Load the trained Random Forest model
@st.cache_resource
def load_model():
    try:
        return joblib.load("random_forest_tgr.pkl")
    except FileNotFoundError:
        st.error("Model file 'random_forest_tgr.pkl' not found. Please ensure it's in the same directory.")
        st.stop()

model = load_model()

# Function to convert SMILES to fingerprint
def smiles_to_fp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)).reshape(1, -1)
        else:
            return None
    except Exception as e:
        st.error(f"Error processing SMILES: {str(e)}")
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
            30% {
                transform: scale(1.02);
            }
            100% { 
                transform: scale(1.15) translateY(-20px);
                opacity: 0;
            }
        }

        @keyframes crackLines {
            0% { 
                opacity: 0;
                transform: scale(1);
            }
            30% { 
                opacity: 1;
                transform: scale(1);
            }
            70% { 
                opacity: 0.8;
                transform: scale(1.1);
            }
            100% { 
                opacity: 0;
                transform: scale(1.2);
            }
        }

        .glass-breaking {
            animation: shatter 1s ease-in-out forwards;
        }

        .glass-breaking::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                linear-gradient(45deg, transparent 48%, rgba(255,255,255,0.9) 49%, rgba(255,255,255,0.9) 51%, transparent 52%),
                linear-gradient(-45deg, transparent 48%, rgba(255,255,255,0.9) 49%, rgba(255,255,255,0.9) 51%, transparent 52%),
                linear-gradient(135deg, transparent 48%, rgba(255,255,255,0.7) 49%, rgba(255,255,255,0.7) 51%, transparent 52%),
                linear-gradient(20deg, transparent 48%, rgba(255,255,255,0.6) 49%, rgba(255,255,255,0.6) 51%, transparent 52%);
            animation: crackLines 1s ease-in-out forwards;
            z-index: 10;
            pointer-events: none;
        }

        .glass-breaking::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            animation: fadeOut 1s ease-in-out forwards;
        }

        @keyframes fadeOut {
            0% { opacity: 1; }
            90% { opacity: 0.1; }
            100% { 
                opacity: 0;
                visibility: hidden;
                display: none;
            }
        }

        /* Fade out glass smoothly */
        .glass-hidden {
            display: none;
        }

        /* Result reveal animation */
        @keyframes resultReveal {
            0% { 
                transform: scale(0.8) translateY(30px);
                opacity: 0;
            }
            70% {
                transform: scale(1.05) translateY(-5px);
                opacity: 1;
            }
            100% { 
                transform: scale(1) translateY(0);
                opacity: 1;
            }
        }

        .result-reveal {
            animation: resultReveal 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
            opacity: 0;
        }

        /* Result container to replace glass */
        .result-container {
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
if 'breaking' not in st.session_state:
    st.session_state.breaking = False
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
if 'result_text' not in st.session_state:
    st.session_state.result_text = ""
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False

# --- MAIN UI ---
# Determine if we should show the glass container
show_glass = not st.session_state.show_result and not st.session_state.breaking

# Determine glass container classes
glass_classes = "glass-container"
if st.session_state.breaking:
    glass_classes = "glass-container glass-breaking"

# Create container - show during initial state and processing, hide during breaking and result
if show_glass or st.session_state.breaking:
    # Add fill animation div
    fill_div = '<div class="chemical-fill"></div>' if st.session_state.processing else ''
    
    st.markdown(f"""
        <div class='{glass_classes}' id='glassContainer'>
            {fill_div}
            <div class='content-layer'>
                <div style='font-size: 2.2em; font-weight: 600; color: #00e0ff; margin-bottom: 10px;'>
                    🧪 TGR Activity Predictor
                </div>
                <div style='font-size: 1.1em; color: #cceeff; margin-bottom: 30px;'>
                    Predict whether a compound is <b>Active</b> or <b>Inactive</b> against Thioredoxin Glutathione Reductase (TGR).
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

if not st.session_state.show_result:
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Input and prediction section
# Only show input if not showing result
if not st.session_state.show_result and not st.session_state.breaking:
    user_input = st.text_input("👉 Enter SMILES:", "", key="smiles_input")

    if st.button("Predict", key="predict_button"):
        if not user_input.strip():
            st.error("⚠️ Please enter a SMILES string.")
        else:
            # Start processing - trigger fill animation
            st.session_state.processing = True
            st.session_state.show_result = False
            st.session_state.breaking = False
            st.session_state.prediction_done = False
            st.session_state.user_smiles = user_input
            st.rerun()

# Handle processing state
if st.session_state.processing and not st.session_state.prediction_done:
    # Show processing message
    st.markdown('<div class="processing-text">⚗️ Analyzing molecular structure...</div>', unsafe_allow_html=True)
    
    # Simulate processing time for animation
    time.sleep(2.5)
    
    # Perform prediction
    fp = smiles_to_fp(st.session_state.user_smiles)
    if fp is None:
        st.session_state.result_text = "error"
    else:
        try:
            prediction = model.predict(fp)[0]
            activity = "🟢 Active" if prediction == 1 else "🔴 Inactive"
            st.session_state.result_text = activity
        except Exception as e:
            st.session_state.result_text = "error"
            st.error(f"Prediction error: {str(e)}")
    
    st.session_state.processing = False
    st.session_state.prediction_done = True
    st.session_state.breaking = True
    st.rerun()

# Handle breaking animation
if st.session_state.breaking and not st.session_state.show_result:
    # Wait for break animation to complete
    time.sleep(1.1)
    st.session_state.breaking = False
    st.session_state.show_result = True
    st.rerun()

# Show result in place of glass
if st.session_state.show_result and st.session_state.prediction_done:
    # Add JavaScript to ensure glass is completely removed from DOM
    st.markdown("""
        <script>
        const glass = document.getElementById('glassContainer');
        if (glass) {
            glass.style.display = 'none';
        }
        </script>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="result-container result-reveal">', unsafe_allow_html=True)
    
    if st.session_state.result_text == "error":
        st.markdown("""
            <div style='font-size: 3em; margin-bottom: 20px;'>❌</div>
            <div style='font-size: 1.5em; color: #ff6b6b; font-weight: 600; margin-bottom: 20px;'>
                Invalid SMILES String
            </div>
            <div style='font-size: 1.1em; color: #ffcccc; margin-bottom: 30px;'>
                Please check your input and try again.
            </div>
        """, unsafe_allow_html=True)
    else:
        # Determine if active or inactive
        is_active = "Active" in st.session_state.result_text
        emoji = "🟢" if is_active else "🔴"
        color = "#00ff88" if is_active else "#ff6b6b"
        status = "Active" if is_active else "Inactive"
        
        st.markdown(f"""
            <div style='font-size: 4em; margin-bottom: 20px;'>{emoji}</div>
            <div style='font-size: 1.3em; color: #cceeff; margin-bottom: 15px;'>
                Prediction Result
            </div>
            <div style='font-size: 2.5em; color: {color}; font-weight: 700; margin-bottom: 30px;'>
                {status}
            </div>
            <div style='font-size: 1em; color: #99ccee; margin-bottom: 20px;'>
                The compound is predicted to be <b>{status.lower()}</b> against<br/>
                Thioredoxin Glutathione Reductase (TGR)
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    # Reset button
    if st.button("🔄 Predict Another", key="reset_button"):
        st.session_state.processing = False
        st.session_state.breaking = False
        st.session_state.show_result = False
        st.session_state.result_text = ""
        st.session_state.prediction_done = False
        if 'user_smiles' in st.session_state:
            del st.session_state.user_smiles
        st.rerun()

# --- Floating Chatbot Icon ---
st.markdown("""
    <div class="chat-icon" title="Open Chat">
        💬
    </div>
""", unsafe_allow_html=True)
