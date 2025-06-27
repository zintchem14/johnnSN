import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from branca.colormap import linear
import time
import base64
from streamlit_autorefresh import st_autorefresh
import requests
from datetime import datetime
import joblib
from typing import Generator
from groq import Groq

# -------- CONFIGURATION DE LA PAGE ----------
st.set_page_config(page_title="Carte Interactive India", layout="wide")



st.markdown("""
    <style>
    section[data-testid="stSidebar"] label {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)







st.markdown("""

<style>
.navbar {
    position: fixed;
    top: 30px;
    left: 0;
    width: 100%;
    background-color: #041E42;
    padding: 10px;
    z-index: 9999;
    display: flex;
    justify-content: center;
    gap: 30px;
    height: 97px;
    color: white;
}
.navbar a {
    color: white;
    font-weight: bold;
    text-decoration: none;
    padding-top: 5px;
    padding-bottom: 35px;
    margin-top: 35px;
}
.navbar a:hover {
    text-decoration: underline;
}
.spacer {
    margin-top: 100px; /* pour éviter que la navbar cache le haut des sections */
}
</style>

<div class="navbar">
    <a href="#donnees-filtrees">Données</a>
    <a href="#seaborn">Graphiques Seaborn</a>
    <a href="#powerbi">Dashboards Power BI</a>
    <a href="#titre">Titre et Modèle</a>
</div>

<div class="spacer"></div>
""", unsafe_allow_html=True)














# -------- IMAGE AVATAR --------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

image_path = "src/JOE.jpg"
image_data = get_base64_image(image_path)

# -------- CSS AVATAR ET SIDEBAR --------


st.markdown(f"""
<style>
section[data-testid="stSidebar"] {{
    background-color: #00333d;
    color: white !important;
}}

.sidebar-content {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-top: 60px;
}}

.avatar-container {{
    width: 240px;
    height: 240px;
    border-radius: 50%;
    border: 6px solid white;
    overflow: hidden;
    animation: spin3D 8s ease-in-out infinite;
}}

.avatar-container img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
}}

.name {{
    margin-top: 20px;
    font-size: 28px;
    color: white;
    font-weight: bold;
    text-align: center;
}}

@keyframes spin3D {{
    0%   {{ transform: rotateY(0deg) rotateX(0deg); }}
    40%  {{ transform: rotateY(180deg) rotateX(30deg); }}
    80%  {{ transform: rotateY(360deg) rotateX(0deg); }}
    100% {{ transform: rotateY(360deg) rotateX(0deg); }}
}}

.pulse-wrapper {{
    position: fixed;
    top: 10%;
    right: 20px;
    transform: translateY(-50%);
    z-index: 999;
}}

.pulse-circle {{
    background-color: #00333d;
    border-radius: 50%;
    width: 226px;
    height: 226px;
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    position: relative;
    text-align: center;
    flex-direction: column;
}}

.pulse-circle .temperature {{
    font-size: 27px;
    margin-bottom: 6px;
}}

.pulse-circle .location {{
    font-size: 20px;
    margin-top: 6px;
}}

.pulse-circle .date {{
    font-size: 29px;
    margin-bottom: 5px;
}}

.pulse-ring {{
    position: absolute;
    top: 0;
    left: 0;
    width: 226px;
    height: 226px;
    border-radius: 50%;
    background-color: #40E0D0;
    opacity: 0.4;
    z-index: 1;
    animation: pulse 2s infinite;
}}

@keyframes pulse {{
    0% {{ transform: scale(1); opacity: 0.5; }}
    70% {{ transform: scale(1.6); opacity: 0; }}
    100% {{ transform: scale(1); opacity: 0; }}
}}
</style>
""", unsafe_allow_html=True)

# -------- AFFICHAGE AVATAR SIDEBAR --------
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-content">
        <div class="avatar-container">
            <img src="data:image/png;base64,{image_data}" alt="Avatar">
        </div>
        <div class="name">JOHANN IABD</div>
    </div>
    """, unsafe_allow_html=True)

# -------- Titre --------
st.title("PRESENTATION DE MON APPLICATION")

# --------- DONNÉES METEO ---------
st_autorefresh(interval=3600000, key="auto_refresh")

API_KEY = "4997fb908b935664f8b5234881c8145f"
lat, lon = 3.848, 11.502
weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}"

response = requests.get(weather_url)
if response.status_code == 200:
    data = response.json()
    temperature = f"{data['main']['temp'] - 273.15:.2f}°C"
    location = data['name']
else:
    temperature = "N/A"
    location = "Inconnue"

date_now = datetime.now().strftime("%a. %d %b").upper()

# -------- AFFICHAGE METEO --------
st.markdown(f"""
<div class="pulse-wrapper">
    <div class="pulse-ring"></div>
    <div class="pulse-circle">
        <div class="temperature">{temperature}</div>
        <div class="date">{date_now}</div>
        <div class="location">{location}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------- CHARGEMENT DES DONNÉES --------
@st.cache_data
def load_data():
    return pd.read_csv("src/df_final.csv")

df = load_data()

# -------- BARRE LATÉRALE DE FILTRES --------
st.sidebar.header("Filtres")

months_loan_duration = st.sidebar.slider("Durée du prêt (mois)", int(df["months_loan_duration"].min()), int(df["months_loan_duration"].max()), (int(df["months_loan_duration"].min()), int(df["months_loan_duration"].max())))
age = st.sidebar.slider("Âge", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))
existing_credits = st.sidebar.selectbox("Nombre de crédits existants", sorted(df["existing_credits"].unique()))
default = st.sidebar.selectbox("Défaut de paiement", sorted(df["default"].unique()))
checking_balance = st.sidebar.multiselect("Solde du compte courant (encodé)", sorted(df["checking_balance_encoded"].unique()), default=sorted(df["checking_balance_encoded"].unique()))
savings_balance = st.sidebar.multiselect("Solde du compte épargne (encodé)", sorted(df["savings_balance_encoded"].unique()), default=sorted(df["savings_balance_encoded"].unique()))

# -------- FILTRAGE --------
filtered_df = df[
    (df["months_loan_duration"].between(*months_loan_duration)) &
    (df["age"].between(*age)) &
    (df["existing_credits"] == existing_credits) &
    (df["default"] == default) &
    (df["checking_balance_encoded"].isin(checking_balance)) &
    (df["savings_balance_encoded"].isin(savings_balance))
]

# -------- AFFICHAGE DES DONNÉES --------
st.markdown(
        f"""
        <div id="donnees-filtrees"><h3>📋 DATAFRAME CREDIT</h3></div>
        """,
        unsafe_allow_html=True
    )
st.markdown("### 📊 Données filtrées")
styled_df = filtered_df.reset_index(drop=True).style.set_table_styles(
    [{
        'selector': 'th',
        'props': [('background-color', '#041E42'), ('color', 'white'), ('font-weight', 'bold')]
    }]
)
st.dataframe(filtered_df.style.background_gradient(cmap='Blues'))



def image_to_base64(path):
    with open(path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

# === Chargement des images ===
img1 = image_to_base64("src/G1.PNG")
img2 = image_to_base64("src/G2.PNG")
img3 = image_to_base64("src/G3.PNG")
img4 = image_to_base64("src/G4.PNG")
img5 = image_to_base64("src/POWER BI 1.PNG")
img6 = image_to_base64("src/POWER B2 1.PNG")
img7 = image_to_base64("src/POWER BI 3.PNG")

# === Style CSS pour galerie ===
st.markdown(
    """
    <style>
    .image-scroll-container {
        display: flex;
        overflow-x: auto;
        padding: 10px;
        white-space: nowrap;
        width: 100%;
        height: 450px;
        margin-bottom: 20px;
    }
    .image-scroll-container img {
        width: 800px;
        height: 400px;
        margin-right: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    .image-scroll-container img:hover {
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Bloc 1 ===
st.markdown(
        f"""
        <div id="seaborn"><h3>📊 Graphiques fait avec Seaborn</h3></div>
        """,
        unsafe_allow_html=True
    )
if st.checkbox("UNE PRESENTATION DES GRAPHS FAITS SUR VSCODE AVEC SEABORN", value=True):
    st.markdown(
        f"""
        <div class="image-scroll-container">
            <img src="{img1}" alt="Image 1">
            <img src="{img2}" alt="Image 2">
            <img src="{img3}" alt="Image 3">
            <img src="{img4}" alt="Image 4">
        </div>
        """,
        unsafe_allow_html=True
    )

# === Bloc 2 ===
st.markdown(
        f"""
        <div id="powerbi"><h3>📈 Graphiques fait avec POWER BI</h3></div>
        """,
        unsafe_allow_html=True
    )
if st.checkbox("UN SOUVENIR AVEC LES DASHBOARD POWER BI VOUS POUVEZ NAVIGUER HORIZONTALEMENT", value=True):
    st.markdown(
        f"""

        <div class="image-scroll-container">
            <img src="{img5}" alt="Image 5">
            <img src="{img6}" alt="Image 6">
            <img src="{img7}" alt="Image 7">
        </div>
        """,
        unsafe_allow_html=True
    )


# Chargement du modèle XGBoost déjà entraîné et sauvegardé
model = joblib.load("src/JOHN_XGBOOST.pkl")

st.markdown(
        f"""
        <div id="titre"><h3>🔮 FORMULAIRE DE PREDICTION</h3></div>
        """,
        unsafe_allow_html=True
    )
# Titre de l'application
st.title("💳 Prédiction de la classe du montant de crédit")
st.markdown("""
Ce modèle prédit une **classe de montant (`amount_class`)** en fonction des caractéristiques numériques liées à un crédit.

### 🧮 Classes prédictives :
- **0** : A1 (Très petit) ≤ 1000  
- **1** : A2 (Petit) 1001 à 2500  
- **2** : B1 (Moyen) 2501 à 4000  
- **3** : B2 (Assez grand) 4001 à 6000  
- **4** : C1 (Grand) 6001 à 10000  
- **5** : C2 (Très grand) > 10000
""")

# Formulaire utilisateur
with st.form("formulaire"):
    months_loan_duration = st.number_input("Durée du prêt (mois)", 1, 100, 6)
    installment_rate = st.slider("Taux de mensualité", 1, 4, 4)
    residence_history = st.slider("Historique de résidence", 1, 4, 2)
    age = st.number_input("Âge", 18, 100, 30)
    existing_credits = st.number_input("Crédits existants", 0, 10, 1)
    default = st.selectbox("Défaut de paiement ?", [0, 1])
    dependents = st.slider("Personnes à charge", 0, 10, 1)
    checking_balance_encoded = st.selectbox("Solde compte courant (encodé)", [0, 1, 2, 3])
    savings_balance_encoded = st.selectbox("Solde épargne (encodé)", [0, 1, 2, 3, 4])
    employment_length_encoded = st.selectbox("Ancienneté emploi (encodé)", [0, 1, 2, 3, 4])

    submit = st.form_submit_button("🔎 Prédire la classe du montant")

if submit:
    # Préparer les données dans le même ordre que lors de l'entraînement
    input_data = pd.DataFrame([[
        months_loan_duration,
        installment_rate,
        residence_history,
        age,
        existing_credits,
        default,
        dependents,
        checking_balance_encoded,
        savings_balance_encoded,
        employment_length_encoded
    ]], columns=[
        'months_loan_duration',
        'installment_rate',
        'residence_history',
        'age',
        'existing_credits',
        'default',
        'dependents',
        'checking_balance_encoded',
        'savings_balance_encoded',
        'employment_length_encoded'
    ])

    # Prédiction
    prediction = model.predict(input_data)[0]

    # Dictionnaire des classes explicatives
    classes = {
        0: "A1 (Très petit)",
        1: "A2 (Petit)",
        2: "B1 (Moyen)",
        3: "B2 (Assez grand)",
        4: "C1 (Grand)",
        5: "C2 (Très grand)"
    }

    # Affichage
    st.success(f"✅ Classe prédite du montant : **{classes[prediction]}**")





def icon(emoji: str):
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🏎️")

st.subheader("Groq Chat Streamlit App", divider="rainbow", anchor=False)

# --- Saisie de la clé API Groq via UI ---
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

api_input = st.text_input(
    "Entrez votre clé API Groq (ex: sk-xxxxxx) :",
    type="password",
    value=st.session_state.api_key,
    help="Votre clé API est sécurisée et ne sera pas affichée."
)

if api_input:
    st.session_state.api_key = api_input

# Ne pas continuer sans clé API
if not st.session_state.api_key:
    st.warning("Veuillez saisir votre clé API Groq pour continuer.")
    st.stop()

# Initialisation du client Groq avec la clé entrée
client = Groq(api_key=st.session_state.api_key)

# Initialisation de la session pour l'historique et modèle choisi
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

models = {
    "gemma2-9b-it": {"name": "Gemma2-9b-it", "tokens": 8192, "developer": "Google"},
    "llama-3.3-70b-versatile": {"name": "LLaMA3.3-70b-versatile", "tokens": 128000, "developer": "Meta"},
    "llama-3.1-8b-instant" : {"name": "LLaMA3.1-8b-instant", "tokens": 128000, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}

col1, col2 = st.columns(2)

with col1:
    model_option = st.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=4
    )

if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]

with col2:
    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512,
        max_value=max_tokens_range,
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust max tokens (max {max_tokens_range})"
    )

# Affichage historique messages
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    full_response = ""

    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            max_tokens=max_tokens,
            stream=True
        )

        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            for chunk in chat_responses_generator:
                full_response += chunk
                st.write(chunk)

    except Exception as e:
        st.error(f"Erreur API : {e}", icon="🚨")

    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})

