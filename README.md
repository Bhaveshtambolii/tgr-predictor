<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="static/logo-light.svg">
    <img src="static/logo-dark.svg" alt="TGR Activity Predictor Logo" width="35%">
  </picture>
</div>

<div align="center">
  <h1>TGR Activity Predictor</h1>
  <p><strong>Predict whether a small molecule (SMILES) inhibits Thioredoxin Glutathione Reductase (TGR) from <em>Schistosoma mansoni</em></strong></p>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)  
[![Render](https://img.shields.io/badge/Deploy-Render-blue.svg)](https://render.com/)
[![Streamlit](https://img.shields.io/badge/Deploy-Streamlit-orange.svg)](https://streamlit.io)

---

### This project was developed as part of the:
<div align="center">

<h3>AI-DRIVEN DRUG DISCOVERY BOOT CAMP (AUGUST 4 – 8, 2025) </h3>  
<h3>Organized by:Department of Biotechnology (Industry Institute Collaboration Cell)</h3> 
<h3>National Institute of Technology – Raipur</h3>  

</div>

**Patron:** Prof. N.V. Ramana Rao, Director, NIT Raipur  
**Chairperson:** Prof. S. Sanyal, Chairman, IICC, NIT Raipur  
**Convenor:** Dr. Awanish Kumar, Associate Professor, Department of Biotechnology, NIT Raipur  
**Coordinator:** Dr. Dijendra Nath Roy, Assistant Professor and IICC Coordinator, Department of Biotechnology, NIT Raipur  
## What is this?
This repository contains a machine-learning pipeline and a Streamlit web app that predicts whether a chemical compound (given as a SMILES string) is **Active** or **Inactive** against the enzyme **TGR** (Thioredoxin Glutathione Reductase) — an antiparasitic drug target in *Schistosoma mansoni*.  
The model is trained on qHTS assay data, SMILES are featurized using RDKit (Morgan fingerprints), and a Random Forest classifier is used for prediction.

---

## Key Features
- Featurization: RDKit Morgan fingerprints (SMILES → fingerprint vectors)  
- Model: RandomForestClassifier (saved as `.pkl`)  
- Web UI: Streamlit app for single-SMILES and batch CSV prediction  
- Deployable: Instructions for Hugging Face Spaces, Streamlit Cloud, Render  
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC & confusion matrix

---

## Demo [![Demo](https://img.shields.io/badge/Demo-Visit%20App-brightgreen.svg)](https://tgr-predictor.onrender.com/)
Try a live demo https://tgr-predictor.onrender.com/



---

