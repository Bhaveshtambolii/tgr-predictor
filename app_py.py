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
st.set_page_config(page_title="TGR Activity AI", page_icon="🧪", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        /* General page background and font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
        
        body {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: white;
            font-family: 'Poppins', sans-serif;
        }
        
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        }

        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Hide default spacing */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
        }

        /* Container wrapper for the glass card */
        .glass-container {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 70vh;
            padding: 20px;
        }

        /* Center Card - Glass Effect */
        .main-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 40px 50px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            color: #fff;
            max-width: 650px;
            width: 100%;
        }

        /* Title */
        .title {
            font-size: 2.5em;
            font-weight: 600;
            color: #00e0ff;
            margin-bottom: 15px;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        /* Subtext */
        .subtitle {
            font-size: 1.05em;
            color: #cceeff;
            margin-bottom: 35px;
            text-align: center;
            line-height: 1.6;
        }

        /* Input section */
        .input-section {
            margin: 30px 0;
        }

        /* Input label styling */
        .input-label {
            color: #cceeff;
            font-weight: 500;
            font-size: 1.05em;
            margin-bottom: 10px;
            display: block;
        }

        /* Streamlit input box styling */
        .stTextInput>div>div>input {
            border-radius: 12px;
            background: rgba(255,255,255,0.95);
            color: #1a1a1a;
            border: 2px solid rgba(0, 224, 255, 0.3);
            padding: 12px 15px;
            font-size: 1em;
            transition: all 0.3s ease;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #00e0ff;
            box-shadow: 0 0 15px rgba(0, 224, 255, 0.5);
            background: rgba(255,255,255,1);
        }
        
        /* Input label from Streamlit */
        .stTextInput>label {
            color: #cceeff !important;
            font-weight: 500 !important;
            font-size: 1.05em !important;
            margin-bottom: 8px !important;
        }

        /* Predict button */
        .stButton>button {
            background: linear-gradient(90deg, #00e0ff, #0077ff);
            color: white;
            font-weight: bold;
            font-size: 1.1em;
            border-radius: 12px;
            padding: 0.7em 2em;
            border: none;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 20px;
            cursor: pointer;
        }

        .stButton>button:hover {
            background: linear-gradient(90deg, #0077ff, #00e0ff);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 224, 255, 0.4);
        }
        
        .stButton>button:active {
            transform: translateY(0);
        }
        
        /* Success/Error messages */
        .stSuccess {
            background: rgba(40, 167, 69, 0.2) !important;
            border: 1px solid rgba(40, 167, 69, 0.4) !important;
            border-radius: 12px !important;
            padding: 15px !important;
            margin-top: 20px !important;
            color: #d4edda !important;
        }
        
        .stError {
            background: rgba(220, 53, 69, 0.2) !important;
            border: 1px solid rgba(220, 53, 69, 0.4) !important;
            border-radius: 12px !important;
            padding: 15px !important;
            margin-top: 20px !important;
            color: #f8d7da !important;
        }

        /* Chatbot floating icon */
        .chat-icon {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: linear-gradient(135deg, #0077ff, #00e0ff);
            border-radius: 50%;
            width: 65px;
            height: 65px;
            color: white;
            font-size: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 6px 20px rgba(0, 120, 255, 0.4);
            cursor: pointer;
            z-index: 999;
            transition: all 0.3s ease;
        }

        .chat-icon:hover {
            transform: scale(1.1);
            box-shadow: 0 8px 25px rgba(0, 224, 255, 0.6);
        }
    </style>
""", unsafe_allow_html=True)


# --- MAIN UI USING CONTAINER ---
# Create a container that will hold everything inside the glass card
st.markdown('<div class="glass-container">', unsafe_allow_html=True)
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# Title and subtitle - now these will be inside the card
st.markdown('<div class="title">🧪 TGR Activity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict whether a compound is <b>Active</b> or <b>Inactive</b> against Thioredoxin Glutathione Reductase (TGR).</div>', unsafe_allow_html=True)

# Input field
user_input = st.text_input("👉 Enter SMILES:", "", key="smiles_input")

# Predict button
if st.button("Predict"):
    if not user_input or user_input.strip() == "":
        st.error("⚠️ Please enter a valid SMILES string.")
    else:
        fp = smiles_to_fp(user_input)
        if fp is None:
            st.error("❌ Invalid SMILES string. Please check your input and try again.")
        else:
            prediction = model.predict(fp)[0]
            activity = "🟢 Active" if prediction == 1 else "🔴 Inactive"
            st.success(f"**Prediction:** {activity}")

# Close the containers
st.markdown('</div>', unsafe_allow_html=True)  # Close main-card
st.markdown('</div>', unsafe_allow_html=True)  # Close glass-container

# --- Floating Chatbot Icon ---
st.markdown("""
    <div class="chat-icon" title="Open Chat">
        💬
    </div>
""", unsafe_allow_html=True)
