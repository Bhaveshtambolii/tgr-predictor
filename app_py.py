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
<style>
        /* Reset */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 0;
        }

        /* Page background */
        .app-container {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            padding: 20px;
        }

        /* Centered glass box */
        .main-card {
            background: rgba(255, 255, 255, 0.12);
            border: 2px solid rgba(255, 255, 255, 0.25);
            border-radius: 20px;
            padding: 50px 40px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
            text-align: center;
            backdrop-filter: blur(20px);
            color: #fff;
            width: 90%;
            max-width: 600px;
            margin: auto;
        }

        /* Title */
        .title {
            font-size: 2.4em;
            font-weight: 700;
            color: #00e0ff;
            margin-bottom: 10px;
        }

        /* Subtext */
        .subtitle {
            font-size: 1.1em;
            color: #d9f2ff;
            margin-bottom: 35px;
        }

        /* Input label */
        .input-label {
            display: block;
            text-align: left;
            margin-bottom: 8px;
            font-size: 1em;
            color: #fff;
        }

        /* Input box */
        .input-box {
            width: 100%;
            border-radius: 10px;
            background: rgba(255,255,255,0.95);
            color: black;
            font-size: 1.05em;
            padding: 10px;
            border: none;
            margin-bottom: 20px;
        }

        .input-box:focus {
            outline: 2px solid #00e0ff;
        }

        /* Predict button */
        .predict-button {
            background: linear-gradient(90deg, #00e0ff, #0077ff);
            color: white;
            font-weight: bold;
            border-radius: 12px;
            padding: 0.6em 1.4em;
            border: none;
            transition: 0.3s;
            font-size: 1em;
            cursor: pointer;
            margin-top: 10px;
        }

        .predict-button:hover {
            background: linear-gradient(90deg, #0077ff, #00e0ff);
            transform: scale(1.05);
        }

        /* Result box */
        .result-box {
            margin-top: 25px;
            padding: 15px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .result-active {
            color: #4cff4c;
            font-size: 1.2em;
            font-weight: bold;
        }

        .result-inactive {
            color: #ff4c4c;
            font-size: 1.2em;
            font-weight: bold;
        }

        /* Floating chat icon */
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
# --- MAIN UI ---
st.markdown("""
    <div class='app-container'>
        <div class='main-card'>
            <div class='title'>🧪 TGR Activity Predictor</div>
            <div class='subtitle'>
                Predict whether a compound is <b>Active</b> or <b>Inactive</b> against Thioredoxin Glutathione Reductase (TGR).
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Input label
st.markdown("<label class='input-label'>👉 Enter SMILES:</label>", unsafe_allow_html=True)

# Text input
user_input = st.text_input("", "", label_visibility="collapsed", key="smilesInput")

# Predict button
if st.button("Predict", key="predict_button"):
    if not user_input:
        st.error("Please enter a SMILES string.")
    else:
        fp = smiles_to_fp(user_input)
        if fp is None:
            st.markdown("""
                <div class='result-box' style='display: block; background-color: #fee; border-left: 4px solid #f44;'>
                    <div style='color: #c33;'>❌ Invalid SMILES string. Please try again.</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            prediction = model.predict(fp)[0]
            activity = "🟢 Active" if prediction == 1 else "🔴 Inactive"
            bg_color = "#e8f5e9" if prediction == 1 else "#ffebee"
            border_color = "#4caf50" if prediction == 1 else "#f44336"
            
            st.markdown(f"""
                <div class='result-box' style='display: block; background-color: {bg_color}; border-left: 4px solid {border_color};'>
                    <div id='resultText' style='font-size: 1.2em; font-weight: bold;'>Prediction: {activity}</div>
                </div>
            """, unsafe_allow_html=True)

# Floating Chat Icon
st.markdown("""
<div class="chat-icon" title="Open Chat">
    💬
</div>
""", unsafe_allow_html=True)
