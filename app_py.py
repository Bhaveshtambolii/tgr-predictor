import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Load model
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
    /* Background and text */
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        font-family: 'Poppins', sans-serif;
        margin: 0;
        padding: 0;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding-top: 0 !important;
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }

    /* Main card container */
    .main-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 3rem auto;
        box-shadow: 0 4px 25px rgba(0, 0, 0, 0.4);
        text-align: center;
        backdrop-filter: blur(12px);
        color: #fff;
        max-width: 600px;
        width: 90%;
    }

    /* Title & subtitle */
    .title {
        font-size: 2em;
        font-weight: 600;
        color: #00e0ff;
        margin-bottom: 0.4em;
        word-wrap: break-word;
        line-height: 1.2em;
    }
    .subtitle {
        font-size: 1em;
        color: #cceeff;
        margin-bottom: 1.8em;
        padding: 0 0.5rem;
    }

    /* Input box styling */
    .stTextInput>div>div>input {
        border-radius: 10px;
        background: rgba(255,255,255,0.9);
        color: black;
        font-size: 1rem;
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
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0077ff, #00e0ff);
        transform: scale(1.05);
    }

    /* Chat icon floating */
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
    .chat-icon:hover { transform: scale(1.1); }

    /* Responsive fix for small screens */
    @media (max-width: 600px) {
        .main-card {
            padding: 1.5rem;
            margin: 1.5rem;
            width: 90%;
        }
        .title {
            font-size: 1.6em;
        }
        .subtitle {
            font-size: 0.95em;
        }
    }
</style>
""", unsafe_allow_html=True)


# --- MAIN UI ---
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<div class='title'>🧪 TGR Activity Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict whether a compound is <b>Active</b> or <b>Inactive</b> against Thioredoxin Glutathione Reductase (TGR).</div>", unsafe_allow_html=True)

user_input = st.text_input("👉 Enter SMILES:", "")

if st.button("Predict"):
    fp = smiles_to_fp(user_input)
    if fp is None:
        st.error("Invalid SMILES string. Please try again.")
    else:
        prediction = model.predict(fp)[0]
        activity = "🟢 Active" if prediction == 1 else "🔴 Inactive"
        st.success(f"Prediction: **{activity}**")

st.markdown("</div>", unsafe_allow_html=True)

# Floating Chat Icon
st.markdown("""
<div class="chat-icon" title="Open Chat">
    💬
</div>
""", unsafe_allow_html=True)        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }

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

        /* Title */
        .title {
            font-size: 2.2em;
            font-weight: 600;
            color: #00e0ff;
            margin-bottom: 10px;
        }

        /* Subtext */
        .subtitle {
            font-size: 1.1em;
            color: #cceeff;
            margin-bottom: 30px;
        }

        /* Input box */
        .stTextInput>div>div>input {
            border-radius: 10px;
            background: rgba(255,255,255,0.9);
            color: black;
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
        }

        .stButton>button:hover {
            background: linear-gradient(90deg, #0077ff, #00e0ff);
            transform: scale(1.05);
        }

        /* Chatbot floating icon */
        .chat-icon {
            position: fixed;
            bottom: 25px;
            right: 25px;
            background: linear-gradient(135deg, #0077ff, #00e0ff);
            border-radius: 50%;
import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Load model
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
        /* Background and text */
        body {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: white;
            font-family: 'Poppins', sans-serif;
            margin: 0;
            padding: 0;
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            padding-top: 0 !important;
        }
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }

        /* Main card container */
        .main-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2rem;
            margin: 3rem auto;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.4);
            text-align: center;
            backdrop-filter: blur(12px);
            color: #fff;
            max-width: 600px;
            width: 90%;
        }

        /* Title & subtitle */
        .title {
            font-size: 2em;
            font-weight: 600;
            color: #00e0ff;
            margin-bottom: 0.4em;
            word-wrap: break-word;
            line-height: 1.2em;
        }
        .subtitle {
            font-size: 1em;
            color: #cceeff;
            margin-bottom: 1.8em;
            padding: 0 0.5rem;
        }

        /* Input box styling */
        .stTextInput>div>div>input {
            border-radius: 10px;
            background: rgba(255,255,255,0.9);
            color: black;
            font-size: 1rem;
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
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #0077ff, #00e0ff);
            transform: scale(1.05);
        }

        /* Chat icon floating */
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
        .chat-icon:hover { transform: scale(1.1); }

        /* Responsive fix for small screens */
        @media (max-width: 600px) {
            .main-card {
                padding: 1.5rem;
                margin: 1.5rem;
                width: 90%;
            }
            .title {
                font-size: 1.6em;
            }
            .subtitle {
                font-size: 0.95em;
            }
        }
    </style>
""", unsafe_allow_html=True)


# --- MAIN UI ---
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<div class='title'>🧪 TGR Activity Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict whether a compound is <b>Active</b> or <b>Inactive</b> against Thioredoxin Glutathione Reductase (TGR).</div>", unsafe_allow_html=True)

user_input = st.text_input("👉 Enter SMILES:", "")

if st.button("Predict"):
    fp = smiles_to_fp(user_input)
    if fp is None:
        st.error("Invalid SMILES string. Please try again.")
    else:
        prediction = model.predict(fp)[0]
        activity = "🟢 Active" if prediction == 1 else "🔴 Inactive"
        st.success(f"Prediction: **{activity}**")

st.markdown("</div>", unsafe_allow_html=True)

# Floating Chat Icon
st.markdown("""
    <div class="chat-icon" title="Open Chat">
        💬
    </div>
""", unsafe_allow_html=True)
