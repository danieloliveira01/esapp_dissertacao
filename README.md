
# Estimativa Automática de Histórias de Usuário 📊

Este projeto é uma aplicação desenvolvida em **Streamlit** que utiliza técnicas de **Machine Learning (ML)** e **Processamento de Linguagem Natural (PLN)** para estimar o esforço necessário para requisitos do tipo *histórias de usuário*.

---

## 🚀 Funcionalidades

- Entrada de histórias de usuário via texto ou upload de arquivos.
- Estimativa automática baseada em um modelo pré-treinado.
- Correção de estimativas diretamente na interface.
- Exportação de resultados corrigidos.

---

## 🛠️ Tecnologias Utilizadas

- Python 3.8+
- Streamlit
- Pandas
- Numpy
- Scikit-learn
- XLNET 
- Transformers (Hugging Face)
- TensorFlow e Keras
- PyTorch 
- huggingface_hub

---

## 📦 Instalação e Execução

### ✅ Pré-requisitos

Certifique-se de ter instalado:

- **Python 3.8 ou superior**
- **pip** (gerenciador de pacotes do Python)
- **Git** (opcional, para clonar o repositório)

#### 🔧 Como instalar o Python e o pip

1. **Acesse o site oficial do Python:**  
   👉 https://www.python.org/downloads/

2. **Baixe o instalador** adequado para seu sistema operacional (Windows, macOS ou Linux).

3. **Durante a instalação no Windows**, marque a opção:  
   ```
   Add Python to PATH
   ```  
   e selecione a opção **"Install Now"**.

4. O `pip` já será instalado automaticamente junto com o Python a partir da versão 3.4+. Você pode verificar se tudo foi instalado corretamente com os comandos:

   ```bash
   python --version
   pip --version
   ```

   Se estiver usando macOS ou Linux, você pode instalar via terminal:

   ```bash
   sudo apt update
   sudo apt install python3 python3-pip   # (Debian/Ubuntu)
   ```

   ou

   ```bash
   brew install python   # (macOS com Homebrew)
   ```

---

### 📥 Passo 1: Clonar o Repositório

```bash
git clone https://github.com/danieloliveira01/esapp_dissertacao.git
cd esapp_dissertacao
```

---

### 💻 Passo 2: Criar um Ambiente Virtual (opcional, mas recomendado)

```bash
python -m venv venv
source venv/bin/activate   # Linux/MacOS
venv\Scripts\activate      # Windows
```

---

### 📦 Passo 3: Instalar Dependências

```bash
pip install -r requirements.txt
```

---

### ▶️ Passo 4: Executar a Aplicação

#### 🔹 Opção A: Usando o script `run.sh` (Linux/macOS)

Execute o script abaixo para automatizar a criação do ambiente virtual, a instalação das dependências e a execução da aplicação:

```bash
chmod +x run.sh
./run.sh
```

#### 🔹 Opção B: Manualmente

```bash
# Ative o ambiente virtual
source venv/bin/activate   # Linux/MacOS
venv\Scripts\activate      # Windows

# Execute a aplicação
streamlit run App.py
```

A aplicação será aberta automaticamente em seu navegador padrão. Caso isso não aconteça, acesse manualmente:  
**http://localhost:8501**

---

## 📂 Estrutura do Projeto

```
├── App.py                     # Arquivo principal da aplicação Streamlit
├── data/                      # Diretório de datasets
├── model/                     # Diretório de arquivos utilitários aos modelos
├── requirements.txt           # Lista de dependências do Python
├── run.sh                     # Script para execução automatizada
├── README.md                  # Este arquivo
```

---

## ⚠️ Observações

- **Modelo Pré-treinado:** Os modelos foram inseridos na plataforma HuggingFace, no repositório DanielOS/ESApp-models.
- **Ambiente Local:** Esta aplicação foi projetada para uso local, mas pode ser adaptada para implantação em serviços como Heroku, AWS, etc.

---

## 💡 Contribuições

Contribuições são bem-vindas!  
Sinta-se à vontade para abrir issues ou enviar pull requests para melhorias e novas funcionalidades.

---

## 📄 Licença

Este projeto está licenciado sob os termos da **MIT License**.
