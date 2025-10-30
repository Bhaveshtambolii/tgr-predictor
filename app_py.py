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

        /* Center Card */
        .main-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.4);
            text-align: center;
            backdrop-filter: blur(12px);
            color: #fff;
            max-width: 600px;
            margin: 50px auto;
        }

        /* Title - now inside the card */
        .title {
            font-size: 2.2em;
            font-weight: 600;
            color: #00e0ff;
            margin-bottom: 10px;
            text-align: center;
        }

        /* Subtext */
        .subtitle {
            font-size: 1.1em;
            color: #cceeff;
            margin-bottom: 30px;
            text-align: center;
        }

        /* Input box styling */
        .stTextInput>div>div>input {
            border-radius: 10px;
            background: rgba(255,255,255,0.9);
            color: black;
            border: 2px solid rgba(0, 224, 255, 0.3);
            padding: 10px;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #00e0ff;
            box-shadow: 0 0 10px rgba(0, 224, 255, 0.5);
        }
        
        /* Input label */
        .stTextInput>label {
            color: #cceeff !important;
            font-weight: 500;
        }

        /* Predict button */
        .stButton>button {
            background: linear-gradient(90deg, #00e0ff, #0077ff);
            color: white;
            font-weight: bold;
            border-radius: 12px;
            padding: 0.6em 1.4em;
            border: none;
            transition: 0.3s;
            width: 100%;
            margin-top: 10px;
        }

        .stButton>button:hover {
            background: linear-gradient(90deg, #0077ff, #00e0ff);
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0, 224, 255, 0.4);
        }
        
        /* Success/Error messages */
        .stSuccess, .stError {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
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
# Opening the card container
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

# Title and subtitle inside the card
st.markdown("<div class='title'>🧪 TGR Activity Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict whether a compound is <b>Active</b> or <b>Inactive</b> against Thioredoxin Glutathione Reductase (TGR).</div>", unsafe_allow_html=True)

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

# Closing the card container
st.markdown("</div>", unsafe_allow_html=True)

# --- Floating Chatbot Icon ---
st.markdown("""
    <div class="chat-icon" title="Open Chat">
        💬
    </div>
""", unsafe_allow_html=True)
