import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

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
st.set_page_config(page_title="TGR Activity AI", page_icon="🧪", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        
        /* ============================================
           RESET & BASE STYLES
           ============================================ */
        
        * {
            box-sizing: border-box;
        }
        
        html, body {
            margin: 0;
            padding: 0;
            overflow-x: hidden;
        }
        
        /* Remove all default Streamlit padding and margins */
        .main .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        /* Full page background with gradient */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            min-height: 100vh;
        }
        
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        
        /* ============================================
           RESPONSIVE LAYOUT CONTAINER
           ============================================ */
        
        .app-wrapper {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: clamp(20px, 4vw, 60px) clamp(15px, 5vw, 40px);
            font-family: 'Poppins', sans-serif;
        }
        
        
        /* ============================================
           GLASS CARD - RESPONSIVE
           ============================================ */
        
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-radius: clamp(15px, 3vw, 25px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            padding: clamp(30px, 5vw, 60px) clamp(25px, 5vw, 60px);
            width: 100%;
            max-width: min(700px, 90vw);
            margin: 0 auto;
        }
        
        
        /* ============================================
           TYPOGRAPHY - FLUID & RESPONSIVE
           ============================================ */
        
        .app-title {
            font-size: clamp(1.8rem, 5vw, 2.8rem);
            font-weight: 700;
            color: #00e0ff;
            text-align: center;
            margin-bottom: clamp(12px, 2vh, 20px);
            text-shadow: 0 0 20px rgba(0, 224, 255, 0.5);
            line-height: 1.2;
            word-wrap: break-word;
        }
        
        .app-subtitle {
            font-size: clamp(0.95rem, 2.5vw, 1.15rem);
            color: #cceeff;
            text-align: center;
            line-height: 1.6;
            margin-bottom: clamp(25px, 4vh, 40px);
            word-wrap: break-word;
        }
        
        
        /* ============================================
           SPACING CONTAINERS
           ============================================ */
        
        .input-container {
            margin: clamp(20px, 3vh, 30px) 0;
        }
        
        .button-container {
            margin: clamp(20px, 3vh, 30px) 0;
        }
        
        .result-container {
            margin-top: clamp(20px, 3vh, 30px);
            text-align: center;
        }
        
        
        /* ============================================
           INPUT FIELD STYLING - RESPONSIVE
           ============================================ */
        
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.95);
            border: 2px solid rgba(0, 224, 255, 0.4);
            border-radius: clamp(10px, 2vw, 15px);
            padding: clamp(12px, 2.5vw, 16px) clamp(15px, 3vw, 20px);
            font-size: clamp(0.9rem, 2vw, 1.05rem);
            color: #1a1a1a;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stTextInput > div > div > input:focus {
            background: rgba(255, 255, 255, 1);
            border-color: #00e0ff;
            box-shadow: 0 0 20px rgba(0, 224, 255, 0.6);
            outline: none;
        }
        
        .stTextInput > div > div > input::placeholder {
            color: #999;
            font-size: clamp(0.85rem, 1.8vw, 0.95rem);
        }
        
        .stTextInput label {
            color: #cceeff !important;
            font-weight: 600 !important;
            font-size: clamp(0.95rem, 2.2vw, 1.1rem) !important;
            margin-bottom: clamp(8px, 1.5vh, 12px) !important;
            display: block;
        }
        
        
        /* ============================================
           BUTTON STYLING - RESPONSIVE
           ============================================ */
        
        .stButton > button {
            background: linear-gradient(90deg, #00e0ff, #0077ff);
            color: white;
            font-weight: 700;
            font-size: clamp(0.95rem, 2.2vw, 1.15rem);
            border: none;
            border-radius: clamp(10px, 2vw, 15px);
            padding: clamp(12px, 2.5vh, 16px) clamp(25px, 5vw, 40px);
            width: 100%;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: clamp(0.5px, 0.15vw, 1px);
            box-shadow: 0 4px 15px rgba(0, 224, 255, 0.3);
        }
        
        .stButton > button:hover {
            background: linear-gradient(90deg, #0077ff, #00e0ff);
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(0, 224, 255, 0.5);
        }
        
        .stButton > button:active {
            transform: translateY(0px);
        }
        
        
        /* ============================================
           MESSAGE STYLING - RESPONSIVE
           ============================================ */
        
        .stSuccess {
            background: rgba(40, 200, 100, 0.15) !important;
            border: 2px solid rgba(40, 200, 100, 0.5) !important;
            border-radius: clamp(10px, 2vw, 15px) !important;
            padding: clamp(15px, 3vw, 22px) !important;
            color: #a8e6cf !important;
            font-size: clamp(1rem, 2.5vw, 1.25rem) !important;
            font-weight: 600 !important;
            text-align: center !important;
            margin-top: clamp(15px, 2vh, 20px) !important;
        }
        
        .stError {
            background: rgba(255, 80, 80, 0.15) !important;
            border: 2px solid rgba(255, 80, 80, 0.5) !important;
            border-radius: clamp(10px, 2vw, 15px) !important;
            padding: clamp(15px, 3vw, 22px) !important;
            color: #ffb3b3 !important;
            font-size: clamp(0.95rem, 2.3vw, 1.15rem) !important;
            font-weight: 600 !important;
            text-align: center !important;
            margin-top: clamp(15px, 2vh, 20px) !important;
        }
        
        .stSpinner > div {
            border-color: #00e0ff !important;
        }
        
        
        /* ============================================
           FLOATING CHATBOT ICON - RESPONSIVE
           ============================================ */
        
        .chat-icon {
            position: fixed;
            bottom: clamp(20px, 3vh, 35px);
            right: clamp(20px, 3vw, 35px);
            background: linear-gradient(135deg, #00e0ff, #0077ff);
            width: clamp(55px, 10vw, 70px);
            height: clamp(55px, 10vw, 70px);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: clamp(24px, 5vw, 32px);
            cursor: pointer;
            box-shadow: 0 6px 24px rgba(0, 224, 255, 0.5);
            transition: all 0.3s ease;
            z-index: 9999;
        }
        
        .chat-icon:hover {
            transform: scale(1.1) rotate(5deg);
            box-shadow: 0 8px 30px rgba(0, 224, 255, 0.7);
        }
        
        
        /* ============================================
           MEDIA QUERIES FOR FINE-TUNED CONTROL
           ============================================ */
        
        /* Small mobile devices */
        @media (max-width: 480px) {
            .app-wrapper {
                padding: 20px 12px;
            }
            
            .glass-card {
                padding: 25px 20px;
                border-radius: 15px;
            }
            
            .app-title {
                font-size: 1.8rem;
                margin-bottom: 12px;
            }
            
            .app-subtitle {
                font-size: 0.95rem;
                margin-bottom: 25px;
            }
            
            .chat-icon {
                bottom: 15px;
                right: 15px;
                width: 55px;
                height: 55px;
                font-size: 24px;
            }
        }
        
        /* Tablets */
        @media (min-width: 481px) and (max-width: 768px) {
            .glass-card {
                max-width: 85vw;
                padding: 35px 30px;
            }
            
            .app-title {
                font-size: 2.2rem;
            }
            
            .app-subtitle {
                font-size: 1.05rem;
            }
        }
        
        /* Small laptops and larger tablets */
        @media (min-width: 769px) and (max-width: 1024px) {
            .glass-card {
                max-width: 650px;
                padding: 45px 45px;
            }
        }
        
        /* Large screens */
        @media (min-width: 1025px) {
            .glass-card {
                max-width: 700px;
                padding: 60px 60px;
            }
            
            .app-title {
                font-size: 2.8rem;
            }
            
            .app-subtitle {
                font-size: 1.15rem;
            }
        }
        
        /* Very large screens */
        @media (min-width: 1440px) {
            .glass-card {
                max-width: 750px;
            }
        }
        
        /* Landscape orientation adjustments */
        @media (max-height: 600px) and (orientation: landscape) {
            .app-wrapper {
                padding: 15px 20px;
            }
            
            .glass-card {
                padding: 20px 30px;
            }
            
            .app-title {
                margin-bottom: 8px;
            }
            
            .app-subtitle {
                margin-bottom: 15px;
            }
            
            .input-container,
            .button-container {
                margin: 15px 0;
            }
        }
    </style>
""", unsafe_allow_html=True)

# --- START OF GLASS CARD ---
st.markdown("""
    <div class="app-wrapper">
        <div class="glass-card">
            <div class="app-title">
                🧪 TGR Activity Predictor
            </div>
            <div class="app-subtitle">
                Predict whether a compound is <strong>Active</strong> or <strong>Inactive</strong> against Thioredoxin Glutathione Reductase (TGR).
            </div>
""", unsafe_allow_html=True)

# Input field with custom styling
st.markdown('<div class="input-container">', unsafe_allow_html=True)

user_input = st.text_input("👉 Enter SMILES:", "", placeholder="e.g., CCO or CC(=O)O")

st.markdown('</div>', unsafe_allow_html=True)

# Button
st.markdown('<div class="button-container">', unsafe_allow_html=True)
predict_button = st.button("🔬 Predict Activity")
st.markdown('</div>', unsafe_allow_html=True)

# Prediction logic
if predict_button:
    if not user_input or user_input.strip() == "":
        st.error("⚠️ Please enter a valid SMILES string.")
    else:
        with st.spinner("Analyzing compound..."):
            fp = smiles_to_fp(user_input)
            if fp is None:
                st.error("❌ Invalid SMILES string. Please check your input and try again.")
            else:
                prediction = model.predict(fp)[0]
                if prediction == 1:
                    st.success("**Prediction: 🟢 Active**")
                else:
                    st.success("**Prediction: 🔴 Inactive**")

# --- END OF GLASS CARD ---
st.markdown("""
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Floating Chatbot Icon ---
st.markdown("""
    <div class="chat-icon" title="Chat Assistant">
        💬
    </div>
""", unsafe_allow_html=True)
