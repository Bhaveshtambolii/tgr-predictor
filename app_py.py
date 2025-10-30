import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Load your model
model = joblib.load("random_forest_tgr.pkl")

# Convert SMILES to fingerprint
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)).reshape(1, -1)
    else:
        return None


# --- PAGE CONFIG ---
st.set_page_config(page_title="TGR Activity Predictor", layout="centered")
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: white;
            font-family: 'Poppins', sans-serif;
        }
        .glass-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.3);
            padding: 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: all 0.6s ease;
        }
        @keyframes smoothShatter {
            0% { transform: scale(1); opacity: 1; filter: brightness(1); }
            40% { transform: scale(1.05); filter: brightness(1.3); }
            70% { transform: scale(1.15) rotate(3deg); opacity: 0.7; }
            100% { transform: scale(1.25) rotate(6deg); opacity: 0; filter: brightness(0.8); }
        }
        .glass-breaking {
            animation: smoothShatter 1.4s ease-in-out forwards;
        }
        @keyframes smoothCrackLines {
            0% { opacity: 0; }
            40% { opacity: 0.7; }
            100% { opacity: 0; }
        }
        .glass-breaking::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background:
                linear-gradient(45deg, transparent 48%, rgba(255,255,255,0.8) 49%, rgba(255,255,255,0.8) 51%, transparent 52%),
                linear-gradient(-45deg, transparent 48%, rgba(255,255,255,0.8) 49%, rgba(255,255,255,0.8) 51%, transparent 52%),
                linear-gradient(135deg, transparent 48%, rgba(255,255,255,0.6) 49%, rgba(255,255,255,0.6) 51%, transparent 52%);
            animation: smoothCrackLines 1s ease-out forwards;
            z-index: 10;
            pointer-events: none;
        }
        .chemical-fill {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 0%;
            background: linear-gradient(180deg, #00e0ff, #007a99);
            animation: fillBeaker 2s ease-in-out forwards;
            border-radius: 0 0 20px 20px;
            z-index: 1;
        }
        @keyframes fillBeaker {
            0% { height: 0%; }
            100% { height: 100%; }
        }
        .content-layer {
            position: relative;
            z-index: 5;
        }
        .result-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 20px;
            backdrop-filter: blur(6px);
            opacity: 0;
            transform: scale(0.8);
            transition: all 0.8s ease;
            z-index: 5;
        }
        .result-overlay.show {
            opacity: 1;
            transform: scale(1);
        }
        .result-text {
            font-size: 2em;
            font-weight: 700;
            color: #ffffff;
            text-shadow: 0 0 10px rgba(0,224,255,0.7);
        }
        .stTextInput > div > div > input {
            background: rgba(255,255,255,0.1);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 10px;
        }
        .stButton>button {
            background: linear-gradient(135deg, #00e0ff, #007a99);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            padding: 0.6em 2em;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 15px #00e0ff;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "processing" not in st.session_state:
    st.session_state.processing = False
if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "result_text" not in st.session_state:
    st.session_state.result_text = ""


# --- GLASS SECTION ---

glass_classes = "glass-container"
if st.session_state.processing:
    glass_classes = "glass-container"
elif st.session_state.show_result:
    glass_classes = "glass-container glass-breaking"

html_content = f"""
<div class="{glass_classes}" id="glassContainer">
    {'<div class="chemical-fill"></div>' if st.session_state.processing else ''}
    <div class="content-layer">
        {""
        if st.session_state.show_result
        else """
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
        """}
    </div>

    {"<div class='result-overlay show'><div class='result-text'>" + st.session_state.result_text + "</div></div>" 
        if st.session_state.show_result and st.session_state.result_text not in ['', 'error'] else ''}
</div>
"""
st.markdown(html_content, unsafe_allow_html=True)


st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)


# --- INPUT AREA ---
smiles_input = st.text_input("Enter SMILES string:", "")

if st.button("Predict Activity"):
    st.session_state.processing = True
    st.session_state.show_result = False
    st.session_state.result_text = ""
    st.rerun()

if st.session_state.processing:
    fp = smiles_to_fp(smiles_input)
    if fp is not None:
        prediction = model.predict(fp)[0]
        label = "🟢 ACTIVE" if prediction == 1 else "🔴 INACTIVE"
        st.session_state.result_text = label
    else:
        st.session_state.result_text = "⚠️ Invalid SMILES"
    st.session_state.processing = False
    st.session_state.show_result = True
    st.rerun()

if st.session_state.show_result and st.button("Predict Another"):
    st.session_state.show_result = False
    st.session_state.result_text = ""
    st.rerun()
