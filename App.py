import streamlit as st
import pandas as pd
import numpy as np
from transformers import XLNetTokenizer, XLNetModel
import torch
#from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from keras.preprocessing.sequence import pad_sequences
import json
import tensorflow as tf
from keras.models import load_model
from huggingface_hub import hf_hub_download

def focal_loss(gamma=2., alpha=None):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])

        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())  # Prevenindo log(0)
        cross_entropy = -y_true * tf.math.log(y_pred)

        if alpha is not None:
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            alpha_factor = y_true * alpha_tensor
            cross_entropy *= alpha_factor

        weight = tf.pow(1. - y_pred, gamma)  # Focal loss weight
        loss = weight * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return focal_loss_fixed
#setando a configuração da pagina
st.set_page_config(page_title="ESApp - Estimador Automático de Pontos de História", layout="wide", page_icon="🎯")



def fibonacci_sequence(n):
    fib_sequence = [0, 1]
    while fib_sequence[-1] < n:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence

def proximo_fibonacci(valor):
    fib_sequence = fibonacci_sequence(valor)
    for fib in fib_sequence:
        if fib >= valor:
            return fib
    return None


# Função para carregar o arquivo de acurácia por classe
def carregar_acuracia_por_classe(lang):
    if lang == "en":
        with open('model/en/acuracia_por_classe_deep_learning.json', 'r') as f:
            acuracia_dict = json.load(f)
        return acuracia_dict
    if lang == "pt":
        with open('model/pt/acuracia_por_classe_deep_learning.json', 'r') as f:
            acuracia_dict = json.load(f)
        return acuracia_dict 


# Função para carregar o modelo XLNet e o tokenizer
def carregar_xlnet_modelo():
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetModel.from_pretrained('xlnet-base-cased')
    return tokenizer, model

# Função para gerar embeddings com o XLNet
def gerar_embeddings_xlnet(texto, tokenizer, model, max_length=512):
    inputs = tokenizer(texto, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    # Pegando o embedding da última camada [CLS] token (ou o embedding médio, se preferir)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings


def estimar_classificacao_deep_learning(lang, texto, modelo_rf, tokenizer, model_xlnet, max_sequence_length=512):
    embeddings = gerar_embeddings_xlnet(texto, tokenizer, model_xlnet, max_length=max_sequence_length)
    # Defina o número de sentenças
    num_sentences = 115
    num_samples = embeddings.shape[0]
    embedding_dim = embeddings.shape[1]
   
    
    
    history = embeddings.reshape(embeddings.shape[0], 1, embeddings.shape[1])
    acuracia_dict = carregar_acuracia_por_classe(lang)
    estimativa = modelo_rf.predict(history)
    story_point = np.argmax(estimativa[0])
    acuracia = acuracia_dict.get(str(story_point), 0)

    if story_point == 0:
         estimative = "Facil"
    elif story_point == 1:
         estimative = "Medio"
    else:
         estimative = "Dificil"

    return estimative, acuracia


# Função de estimativa usando o Random Forest (após a conversão em embeddings)
def estimar_ponto_por_historia_random_forest(lang, texto, modelo_rf, tokenizer, model_xlnet, max_sequence_length=512):
    # Gerando embeddings com XLNet
    embeddings = gerar_embeddings_xlnet(texto, tokenizer, model_xlnet, max_length=max_sequence_length)
    acuracia_dict = carregar_acuracia_por_classe(lang)
    
   
    # Passando os embeddings para o modelo Random Forest para prever o ponto
    estimativa = modelo_rf.predict(embeddings)
    estimativa_fib = proximo_fibonacci(estimativa[0])
     # Obter a acurácia da classe predita
    acuracia = acuracia_dict.get(str(estimativa_fib), 0)
    # Aqui você pode ajustar para estimar com base no valor predito
    return estimativa_fib, acuracia







#Calculando a sequencia de fibonacci 
def fibonacci_sequence(n):
    fib_sequence = [0, 1]
    while fib_sequence[-1] < n:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence
#Estimar os pontos por história com base no resultado do modelo.
def proximo_fibonacci(valor):
    fib_sequence = fibonacci_sequence(valor)
    for fib in fib_sequence:
        if fib >= valor:
            return fib
    return None

# Função para estimar ponto por história, onde recebera como parametro, o modelo compilado, texto, tokenizer e tamanho max.
#tokenizer serve para realizar a tokenização do texto, ou seja, selecionar as palavras de maior relevancia.
def estimar_ponto_por_historia_deep(texto, modelo, tokenizer, max_sequence_length):

    texto = [texto]
    encSequences_test = tokenizer.texts_to_sequences(texto)
    x_test = pad_sequences(encSequences_test, maxlen=max_sequence_length, padding='post')
    previsao = modelo.predict(x_test, batch_size=None, verbose=0, steps=None)
    estimativa = previsao[0][0]
    return estimativa

#Carregando o modelo e o tokenizer
#modelo = joblib.load('model/model_LSTM.joblib')
modelo = ""
df = pd.read_csv('data/reqTxt.csv', header=None)
dfRequire = df.iloc[:, :]
X = dfRequire[0]
X = np.array(X)

MAX_LEN = 23313
tokenizer = Tokenizer(num_words=MAX_LEN, char_level=False, lower=False)
tokenizer.fit_on_texts(X)
MAX_SEQUENCE_LENGTH = 100

st.sidebar.title("Selecione o modelo de estimativa")
option_model = st.sidebar.selectbox("Escolha o método de estimativa do ponto por história:", ("Classificacao - en", "Classificacao - pt", "Numérico - en", "Numérico - pt" ))
lang = ""
if option_model == "Classificacao - en":
     
     model_path = hf_hub_download(repo_id="DanielOS/ESApp-models", filename="en/model_classificacao_v1_05_25_en.h5")
     #modelo = load_model('model/en/model_classificacao_v1_05_25_en.h5', custom_objects={'focal_loss_fixed': focal_loss(gamma=2., alpha=None)})
     modelo = load_model(model_path,custom_objects={'focal_loss_fixed': focal_loss(gamma=2., alpha=None)} )
     #definir modelo random forest
     #modelo = joblib.load('model/en/deep_learning_classificacao.pkl')
     #modelo = load_model('model/en/model_classificacao_v1_05_25_en.h5', custom_objects={'focal_loss_fixed': focal_loss(gamma=2., alpha=None)})
     lang = "en"

if option_model == "Classificacao - pt":
     #definir modelo random forest
     model_path = hf_hub_download(repo_id="DanielOS/ESApp-models", filename="pt/model_classificacao_v1_05_25_pt.h5")
     
     #modelo = load_model('model/pt/model_classificacao_v1_05_25_pt.h5', custom_objects={'focal_loss_fixed': focal_loss(gamma=2., alpha=None)})
     modelo = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss(gamma=2., alpha=None)})
     lang = "pt"
    

if option_model == "Numérico - en":
     #definir modelo random forest
     #modelo = joblib.load('model/en/deep_learning_regressao.pkl')

     model_path = hf_hub_download(repo_id="DanielOS/ESApp-models", filename="en/model_deep_learning_regressao_en.h5")
     modelo = load_model(model_path, custom_objects={
                       'mse': tf.keras.losses.MeanSquaredError()
                   })
     ang = "en"

if option_model == "Numérico - pt":
     #definir modelo random forest
     #modelo = joblib.load('model/pt/deep_learning_pt.pkl')
     model_path = hf_hub_download(repo_id="DanielOS/ESApp-models", filename="pt/model_deep_learning_regressao_pt.h5")
     modelo = load_model(model_path, custom_objects={
                       'mse': tf.keras.losses.MeanSquaredError()
                   })
     lang = "pt"
#Gerando a interface do WebApp com Streamlit
st.title('🎯 ESApp - Estimador Automático de Pontos de História')
st.markdown("""
Bem-vindo ao ESApp! Esta ferramenta ajuda você a estimar pontos de história automaticamente usando um modelo de aprendizado de máquina.
Você pode inserir uma descrição de história de usuário individualmente ou carregar um arquivo CSV com várias histórias.
""")
# Inicializando variáveis no st.session_state
if 'estimativa' not in st.session_state:
    st.session_state.estimativa = None
if 'correcao' not in st.session_state:
    st.session_state.correcao = 0

if 'df_uploaded' not in st.session_state:
    st.session_state.df_uploaded = None
#Sidebar par selecionar a opção de entrada, podendo ser história de usuário unica ou um arquivo csv com várias.
st.sidebar.title("Opções de Entrada")
option = st.sidebar.radio("Escolha o método de entrada:", ("Inserir Manualmente", "Carregar Arquivo CSV"))
if 'estimativa' not in st.session_state:
    st.session_state.estimativa = None
if option == "Inserir Manualmente":
    # Seção para entrada manual de descrição de história
    st.subheader('Inserir História de Usuário Individualmente')
    user_input = st.text_area("Insira a descrição da história:")
    user_input = user_input.replace(",", "")

    if st.button("Estimar Ponto por História"):
        if user_input:
            
            #spinner para carregar enquanto processa.
            if lang == "":
                 lang = "en"
            if "Numérico" in option_model:
                with st.spinner('Estimando ponto por história...'):
                    
                    st.session_state.estimativa = estimar_ponto_por_historia_deep(user_input, modelo, tokenizer, MAX_SEQUENCE_LENGTH)
                    
                st.success(f"Ponto por história sugerido: {st.session_state.estimativa:.1f}")
                if lang == "en":
                        st.info(f"Este modelo possui um erro médio de 3.77 na assertividade")
                if lang == "pt":
                        st.info(f"Este modelo possui um erro médio de 3.94 na assertividade")
               
                
            if "Classificacao" in option_model:
                with st.spinner('Estimando ponto por história...'):
                    tokenizer, model_xlnet = carregar_xlnet_modelo()
                    #st.session_state.estimativa, st.session_state.acuracia = estimar_ponto_por_historia_random_forest(lang, user_input, modelo, tokenizer, model_xlnet)
                    st.session_state.estimativa, st.session_state.acuracia = estimar_classificacao_deep_learning(lang, user_input, modelo, tokenizer, model_xlnet)
                    
                    
                    #st.session_state.estimativa = estimar_ponto_por_historia_deep(user_input, modelo, tokenizer, MAX_SEQUENCE_LENGTH)
                st.success(f"Ponto por história sugerido: {st.session_state.estimativa}")
                #st.info(f"Para esta estimativa, este modelo possui: {st.session_state.acuracia:.1f}% de assertividade")
                st.info(f"Para esta estimativa, este modelo possui: {st.session_state.acuracia:.1f}% de assertividade")
            # Mostrar a estimativa inicial
            #st.success(f"Ponto por história sugerido: {estimativa}")

            # Adicionar campo para o usuário corrigir a estimativa

    # Passo 2: Adicionar Correção Manual e Salvar
            
            st.session_state.correcao = st.number_input("Se a estimativa está incorreta, insira o ponto correto aqui:", min_value=1)

    if st.button("Salvar Correção"):
                        
                        if st.session_state.correcao != st.session_state.estimativa:
                            
                            with open('correcoes.csv', 'a') as f:
                                f.write(f"{user_input},{st.session_state.estimativa},{st.session_state.correcao}\n")
                            st.success("Correção salva com sucesso!")
                        else:
                            st.write(st.session_state.correcao)
                            st.write(st.session_state.estimativa)
                            st.warning("A correção é igual à estimativa original. Não foi necessário salvar.")     
elif option == "Carregar Arquivo CSV":
    #Carregar o arquivo CSV
    st.subheader('Carregar Arquivo CSV com Histórias de Usuários')
    #aqui, explicando como deve ser o arquivo
    st.markdown("""
    **Formato do arquivo CSV:**
    - Deve conter pelo menos três colunas: ID, Título e História.
    """)

    #Gerando um exemplo de arquivo para o usuário fazer download.
    example_csv = pd.DataFrame({
        'ID': [1, 2],
        'Título': ['Login no Sistema', 'Gerenciamento de Usuários'],
        'História': [
            'Como usuária, quero poder fazer login no sistema para acessar minhas informações pessoais.',
            'Como administrador, quero poder gerenciar os usuários para manter o sistema organizado e seguro.'
        ]
    })
    csv = example_csv.to_csv(index=False).encode('utf-8')

    # Botão para download do exemplo de CSV
    st.download_button(
        label="Baixar Exemplo de Arquivo CSV",
        data=csv,
        file_name='exemplo_historias.csv',
        mime='text/csv'
    )

    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if st.button("Estimar Pontos por Histórias"):
        if uploaded_file:
            df_uploaded = pd.read_csv(uploaded_file)
            if len(df_uploaded.columns) < 3:
                st.error("O arquivo CSV deve ter pelo menos três colunas (ID, Título, História).")
            else:
                with st.spinner('Estimando pontos por histórias...'):
                    historias = df_uploaded.iloc[:, 2].tolist()
                    if "Numérico" in option_model:
                        estimativas = [estimar_ponto_por_historia_deep(historia, modelo, tokenizer, MAX_SEQUENCE_LENGTH) for historia in historias]
                        if lang == "en":
                            st.info(f"Este modelo possui um erro médio de 3.77 na assertividade")
                        if lang == "pt":
                            st.info(f"Este modelo possui um erro médio de 3.91 na assertividade")
                        
                        st.session_state.df_uploaded = df_uploaded.copy()
                        st.session_state.df_uploaded['estimativa'] = estimativas
                    if  "Classificacao" in option_model :

                        tokenizer, model_xlnet = carregar_xlnet_modelo()
                        #st.session_state.estimativa, st.session_state.acuracia = estimar_ponto_por_historia_random_forest(lang, user_input, modelo, tokenizer, model_xlnet)
                        #st.session_state.estimativa, st.session_state.acuracia = estimar_classificacao_deep_learning(lang, user_input, modelo, tokenizer, model_xlnet)
                        
                        
                        #tokenizer, model_xlnet = carregar_xlnet_modelo()
                        #resultados = [estimar_ponto_por_historia_random_forest(lang, historia, modelo, tokenizer, model_xlnet) for historia in historias]
                        resultados = [estimar_classificacao_deep_learning(lang, historia, modelo, tokenizer, model_xlnet) for historia in historias]
                        estimativas, acuracias = zip(*resultados)
                        print("estimativa:", estimativas)

        
                        #st.info(f"Para esta estimativa, este modelo possui: {st.session_state.acuracia:.1f}% de assertividade")
                   
                        st.session_state.df_uploaded = df_uploaded.copy()
                        st.session_state.df_uploaded['estimativa'] = estimativas
                        st.session_state.df_uploaded['acuracia'] = acuracias
                   
                    # Inicializando correções com as estimativas iniciais
                    st.session_state.correcoes = list(estimativas).copy()
                st.success("Estimativas geradas com sucesso!")
                st.write(st.session_state.df_uploaded)

                
                # Permitir download do arquivo com estimativas
                csv = st.session_state.df_uploaded.to_csv(index=False).encode('utf-8')
                st.download_button("Baixar CSV com Estimativas", data=csv, file_name="estimativas.csv", mime='text/csv')

                # Opção para correção manual das estimativas
            st.subheader('Correção Manual das Estimativas')
            correcoes = []
        correcoes_estimativas = []
        for i in range(len(st.session_state.df_uploaded)):
            if "Classificacao" in option_model:
                 x = st.selectbox(
                    f"Correção para ID {st.session_state.df_uploaded.iloc[i, 0]} (Estimativa Original: {st.session_state.df_uploaded.iloc[i, -1]}):",
                    options=["Facil", "Medio", "Dificil"],
                    index=["Facil", "Medio", "Dificil"].index(st.session_state.correcoes[i]) if st.session_state.correcoes[i] in ["Facil", "Medio", "Dificil"] else 0,
                    key=f"correcao_{i}"
                )
            else:
                x = st.number_input(f"Correção para ID {st.session_state.df_uploaded.iloc[i, 0]} (Estimativa Original: {st.session_state.df_uploaded.iloc[i, -1]}):", min_value=0, value=int(st.session_state.correcoes[i]), key=f"correcao_{i}")
            st.session_state.correcoes[i] = x
        # Atualizando o DataFrame com as correções
        st.session_state.df_uploaded['Correção'] = st.session_state.correcoes

        # Permitir download do arquivo com estimativas e correções
        csv_corrigido = st.session_state.df_uploaded.to_csv(index=False).encode('utf-8')
        st.download_button("Baixar CSV com Estimativas e Correções", data=csv_corrigido, file_name="estimativas_corrigidas.csv", mime='text/csv')
    
    else:
            st.warning("Por favor, carregue um arquivo CSV contendo as histórias.")

# Adicionar link para documentação ou ajuda adicional
st.sidebar.markdown("Para mais informações, visite a [documentação](https://link_tese_mestrado).")


# Adicionar ícone de ajuda com tooltip sobre o formato do arquivo CSV
if option == "Carregar Arquivo CSV":
    st.sidebar.markdown("""
        <style>
        .info-icon {
            display: inline-flex;
            align-items: center;
        }
        .info-icon img {
            margin-right: 5px;
            
        }
        </style>
        <div class="info-icon">
            <img src="https://img.icons8.com/ios-filled/50/000000/info.png" width="20" height="20">
            <span>Formato do Arquivo CSV</span>
        </div>
        """, unsafe_allow_html=True)
    st.sidebar.info("""
    O arquivo CSV deve conter pelo menos três colunas: ID, Título e História. 

    """)
st.sidebar.markdown("""
---
**Nota:** A estimativa apresentada é baseada em um modelo de aprendizado de máquina para auxiliar o time de desenvolvimento. É recomendada uma análise adicional pelo time para validação e ajustes necessários.
""")
