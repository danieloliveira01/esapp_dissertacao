#!/bin/bash

python3 -m venv venv

# 2. Ative o ambiente virtual
source venv/bin/activate  # No Linux/macOS

echo "Instalando dependências..."
pip install -r requirements.txt

echo "Iniciando a aplicação Streamlit..."
streamlit run App.py

