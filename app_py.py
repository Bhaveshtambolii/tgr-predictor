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

        /* Success/Error messages */
        .stSuccess, .stError {
            background: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 10px !important;
            padding: 15px !important;
            margin-top: 20px !important;
        }

        .stSuccess p, .stError p {
            color: #ffffff !important;
            font-size: 18px !important;
            font-weight: 500 !important;
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
    </style>
""", unsafe_allow_html=True)


# --- MAIN UI ---
# Create a container with custom styling
st.markdown("""
    <div style='
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 4px 25px rgba(0, 0, 0, 0.4);
        text-align: center;
        backdrop-filter: blur(12px);
        max-width: 600px;
        margin: 0 auto;
    '>
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
""", unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Input and prediction section
user_input = st.text_input("👉 Enter SMILES:", "", key="smiles_input")

if st.button("Predict"):
    if not user_input.strip():
        st.error("⚠️ Please enter a SMILES string.")
    else:
        fp = smiles_to_fp(user_input)
        if fp is None:
            st.error("❌ Invalid SMILES string. Please try again.")
        else:
            prediction = model.predict(fp)[0]
            activity = "🟢 Active" if prediction == 1 else "🔴 Inactive"
            st.success(f"Prediction: **{activity}**")

# --- Floating Chatbot Icon ---
st.markdown("""
    <div class="chat-icon" title="Open Chat">
        💬
    </div>
""", unsafe_allow_html=True)
