import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Load trained Random Forest model
model = joblib.load("random_forest_tgr.pkl")

# Convert SMILES to fingerprint
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
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

        body {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            font-family: 'Poppins', sans-serif;
            color: white;
        }

        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        }

        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }

        /* --- Glass Card --- */
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.25);
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            padding: 60px 50px;
            margin: 80px auto;
            text-align: center;
            max-width: 650px;
            animation: fadeIn 1.2s ease forwards;
        }

        /* --- Title --- */
        .title {
            font-size: 2.6rem;
            font-weight: 700;
            color: #00e0ff;
            margin-bottom: 15px;
            text-shadow: 0 0 25px rgba(0, 224, 255, 0.6);
            letter-spacing: 0.5px;
        }

        /* --- Subtitle --- */
        .subtitle {
            font-size: 1.1rem;
            color: #cceeff;
            margin-bottom: 55px;
            line-height: 1.7;
            animation: fadeInUp 1.2s ease;
        }

        /* --- Input field --- */
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 14px 18px;
            font-size: 1rem;
            border: 2px solid rgba(0, 224, 255, 0.4);
            transition: all 0.3s ease;
        }

        .stTextInput > div > div > input:focus {
            border-color: #00e0ff;
            box-shadow: 0 0 20px rgba(0, 224, 255, 0.6);
        }

        /* --- Button --- */
        .stButton > button {
            background: linear-gradient(90deg, #00e0ff, #0077ff);
            color: white;
            font-weight: 700;
            border: none;
            border-radius: 12px;
            padding: 12px 25px;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            margin-top: 40px;
            box-shadow: 0 4px 15px rgba(0, 224, 255, 0.3);
        }

        .stButton > button:hover {
            background: linear-gradient(90deg, #0077ff, #00e0ff);
            transform: translateY(-3px);
            box-shadow: 0 6px 25px rgba(0, 224, 255, 0.5);
        }

        /* --- Success / Error --- */
        .stSuccess {
            background: rgba(40, 200, 100, 0.15) !important;
            border: 2px solid rgba(40, 200, 100, 0.5) !important;
            border-radius: 12px !important;
            padding: 18px !important;
            color: #a8e6cf !important;
            font-weight: 600 !important;
            margin-top: 30px !important;
            text-align: center;
        }

        .stError {
            background: rgba(255, 80, 80, 0.15) !important;
            border: 2px solid rgba(255, 80, 80, 0.5) !important;
            border-radius: 12px !important;
            padding: 18px !important;
            color: #ffb3b3 !important;
            font-weight: 600 !important;
            margin-top: 30px !important;
            text-align: center;
        }

        /* --- Chat Icon --- */
        .chat-icon {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: linear-gradient(135deg, #00e0ff, #0077ff);
            width: 65px;
            height: 65px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 30px;
            box-shadow: 0 6px 25px rgba(0, 224, 255, 0.5);
            transition: all 0.3s ease;
            cursor: pointer;
            z-index: 999;
        }

        .chat-icon:hover {
            transform: scale(1.1) rotate(5deg);
            box-shadow: 0 8px 35px rgba(0, 224, 255, 0.7);
        }

        /* --- Animations --- */
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        @keyframes fadeInUp {
            0% {opacity: 0; transform: translateY(25px);}
            100% {opacity: 1; transform: translateY(0);}
        }
    </style>
""", unsafe_allow_html=True)

# --- MAIN CARD ---
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='title'>🧪 TGR Activity Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict whether a compound is <b>Active</b> or <b>Inactive</b> against Thioredoxin Glutathione Reductase (TGR).</div>", unsafe_allow_html=True)

user_input = st.text_input("👉 Enter SMILES:", "", placeholder="e.g., CCO or CC(=O)O")

if st.button("🔬 Predict Activity"):
    if not user_input or user_input.strip() == "":
        st.error("⚠️ Please enter a valid SMILES string.")
    else:
        fp = smiles_to_fp(user_input)
        if fp is None:
            st.error("❌ Invalid SMILES string. Please check your input.")
        else:
            prediction = model.predict(fp)[0]
            if prediction == 1:
                st.success("**Prediction: 🟢 Active**")
            else:
                st.success("**Prediction: 🔴 Inactive**")

st.markdown("</div>", unsafe_allow_html=True)

# --- Floating Chatbot Icon ---
st.markdown("""
    <div class="chat-icon" title="Chat Assistant">
        💬
    </div>
""", unsafe_allow_html=True)
