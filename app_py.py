import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
lc_api_key = os.getenv("LC_API_KEY")
if lc_api_key:
    os.environ['LANGCHAIN_API_KEY'] = lc_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Chat bot for TGR Activity Predictor"

q_api_key = os.getenv("Q_API_KEY")

# Load the trained Random Forest model
model = joblib.load("random_forest_tgr.pkl")

# Function to convert SMILES to fingerprint
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)).reshape(1, -1)
    else:
        return None

## Prompt template for chatbot
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful assistant for the TGR Activity Predictor application. Your role is to help users understand how to use the predictor, explain features, answer questions about TGR (Thioredoxin Glutathione Reductase), guide them through the prediction process, and provide general assistance. You do not make predictions yourself - the prediction model does that.

Additional capabilities:
1. If a user provides a SMILES notation (like c1ccccc1, CCO, CC(=O)O, etc.), identify and tell them the name of the compound.
2. If a user provides the name of a chemical compound (like benzene, ethanol, aspirin, etc.), generate its SMILES notation.

Please respond to user queries in a friendly and helpful manner."""),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, api_key):
    if not api_key:
        return "⚠️ Please configure your Groq API Key in the settings above to use the chatbot."

    try:
        model_chat = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=api_key
        )
        output_parser = StrOutputParser()
        chain = prompt | model_chat | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        return f"❌ Error: {str(e)}. Please check your API key in settings."


# --- PAGE CONFIG ---
st.set_page_config(page_title="TGR Activity AI", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        /* General page background and font */
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
        
        /* Sidebar styling - Always open with fixed width */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e, #16213e);
            min-width: 400px !important;
            max-width: 400px !important;
            width: 400px !important;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            width: 400px !important;
        }
        
        /* Hide sidebar collapse button */
        [data-testid="collapsedControl"] {
            display: none;
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: white;
        }

        /* Center Card */
        .main-card {
         
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
        
        /* Chat message styling */
        .stChatMessage {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CHATBOT ---
with st.sidebar:
    st.markdown("### 💬 AI Assistant")
    st.markdown("---")

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = None  # No default API key

    # API Key Configuration Section
    with st.expander("🔑 API Key Settings", expanded=not st.session_state.groq_api_key):
        st.markdown("**Groq API Key Configuration**")
        st.caption("The chatbot requires your Groq API key to function.")

        # Show current status
        if st.session_state.groq_api_key:
            st.success("✓ API Key configured - Chatbot is ready!")
        else:
            st.warning("⚠ No API Key set - Chatbot is disabled")

        # API Key input
        api_key_input = st.text_input(
            "Enter your Groq API Key:",
            value="",
            type="password",
            placeholder="gsk_...",
            help="Get your free API key from https://console.groq.com/keys"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Key", use_container_width=True):
                if api_key_input:
                    st.session_state.groq_api_key = api_key_input
                    st.success("API Key saved!")
                    st.rerun()
                else:
                    st.error("Please enter a valid API key")

        with col2:
            if st.button("🗑️ Clear Key", use_container_width=True):
                st.session_state.groq_api_key = None
                st.session_state.messages = []  # Clear chat history too
                st.success("API Key cleared")
                st.rerun()

    # Information section at top
    with st.expander("ℹ️ About Assistant"):
        st.info("""
        **I can help you with:**
        - How to use the predictor
        - Understanding TGR
        - Convert SMILES ↔ Names
        - Interpreting results

        **Examples:**
        - "What is c1ccccc1?"
        - "SMILES for aspirin?"
        - "How does TGR work?"
        """)
    
    st.markdown("---")

    # Check if API key is configured
    if not st.session_state.groq_api_key:
        st.info("💡 **Chatbot is disabled**\n\nPlease add your Groq API key in the settings above to start chatting.")
    else:
        # Display chat history with scrollable container
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat input at bottom (only shown when API key is set)
        question = st.chat_input("Ask me anything...")

        # Process new message
        if question:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})

            # Generate and display assistant response
            with st.spinner("Thinking..."):
                response = generate_response(question, st.session_state.groq_api_key)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        st.markdown("---")

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()


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
