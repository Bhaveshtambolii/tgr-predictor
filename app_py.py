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
        
        /* Remove all default Streamlit padding and margins */
        .main .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        /* Full page background */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        }
        
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Full page wrapper */
        .app-wrapper {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 40px 20px;
            font-family: 'Poppins', sans-serif;
        }
        
        /* Glass card container */
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-radius: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            padding: 50px 60px;
            max-width: 700px;
            width: 100%;
        }
        
        /* Title styling */
        .app-title {
            font-size: 2.8em;
            font-weight: 700;
            color: #00e0ff;
            text-align: center;
            margin-bottom: 15px;
            text-shadow: 0 0 20px rgba(0, 224, 255, 0.5);
        }
        
        /* Subtitle styling */
        .app-subtitle {
            font-size: 1.1em;
            color: #cceeff;
            text-align: center;
            line-height: 1.7;
            margin-bottom: 40px;
        }
        
        /* Input container */
        .input-container {
            margin: 30px 0;
        }
        
        /* Button container */
        .button-container {
            margin: 25px 0;
        }
        
        /* Result container */
        .result-container {
            margin-top: 25px;
            text-align: center;
        }
        
        /* Chatbot icon */
        .chat-icon {
            position: fixed;
            bottom: 35px;
            right: 35px;
            background: linear-gradient(135deg, #00e0ff, #0077ff);
            width: 70px;
            height: 70px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            cursor: pointer;
            box-shadow: 0 6px 24px rgba(0, 224, 255, 0.5);
            transition: all 0.3s ease;
            z-index: 9999;
        }
        
        .chat-icon:hover {
            transform: scale(1.1) rotate(5deg);
            box-shadow: 0 8px 30px rgba(0, 224, 255, 0.7);
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

# Custom CSS for input field to match the design
st.markdown("""
    <style>
        /* Style the input field */
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.95);
            border: 2px solid rgba(0, 224, 255, 0.4);
            border-radius: 15px;
            padding: 15px 20px;
            font-size: 1em;
            color: #1a1a1a;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            background: rgba(255, 255, 255, 1);
            border-color: #00e0ff;
            box-shadow: 0 0 20px rgba(0, 224, 255, 0.6);
            outline: none;
        }
        
        .stTextInput label {
            color: #cceeff !important;
            font-weight: 600 !important;
            font-size: 1.1em !important;
            margin-bottom: 10px !important;
        }
        
        /* Style the button */
        .stButton > button {
            background: linear-gradient(90deg, #00e0ff, #0077ff);
            color: white;
            font-weight: 700;
            font-size: 1.15em;
            border: none;
            border-radius: 15px;
            padding: 15px 40px;
            width: 100%;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stButton > button:hover {
            background: linear-gradient(90deg, #0077ff, #00e0ff);
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 224, 255, 0.5);
        }
        
        .stButton > button:active {
            transform: translateY(-1px);
        }
        
        /* Style success message */
        .stSuccess {
            background: rgba(40, 200, 100, 0.15) !important;
            border: 2px solid rgba(40, 200, 100, 0.5) !important;
            border-radius: 15px !important;
            padding: 20px !important;
            color: #a8e6cf !important;
            font-size: 1.2em !important;
            font-weight: 600 !important;
            text-align: center !important;
        }
        
        /* Style error message */
        .stError {
            background: rgba(255, 80, 80, 0.15) !important;
            border: 2px solid rgba(255, 80, 80, 0.5) !important;
            border-radius: 15px !important;
            padding: 20px !important;
            color: #ffb3b3 !important;
            font-size: 1.1em !important;
            font-weight: 600 !important;
            text-align: center !important;
        }
    </style>
""", unsafe_allow_html=True)

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
