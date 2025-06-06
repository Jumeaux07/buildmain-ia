import streamlit as st
import nltk
import heapq
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from  PIL import Image

nltk.download("punkt")
nltk.download("stopwords")

# ----------------- FONCTIONS -----------------

def resumer_texte(texte, nb_phrases=2):
    phrases = sent_tokenize(texte)
    mots = word_tokenize(texte.lower())
    mots_utiles = [mot for mot in mots if mot.isalnum() and mot not in stopwords.words("french")]
    frequences = Counter(mots_utiles)
    max_freq = max(frequences.values())
    for mot in frequences:
        frequences[mot] /= max_freq

    scores_phrases = {}
    for phrase in phrases:
        for mot in word_tokenize(phrase.lower()):
            if mot in frequences and len(phrase.split(" ")) < 30:
                scores_phrases[phrase] = scores_phrases.get(phrase, 0) + frequences[mot]

    return heapq.nlargest(nb_phrases, scores_phrases, key=scores_phrases.get)

def analyser_frequences(texte):
    mots = word_tokenize(texte.lower())
    mots_utiles = [mot for mot in mots if mot.isalnum() and mot not in stopwords.words("french")]
    return Counter(mots_utiles).most_common(10)

def chatbot_local(message):
    message = message.lower()
    if "bonjour" in message:
        return "Bonjour ! Je suis BuildBot. En quoi puis-je vous aider ?"
    elif "aide" in message or "quoi" in message:
        return "Je peux résumer du texte, analyser les mots ou répondre à des questions simples."
    elif "merci" in message:
        return "Avec plaisir ! 😊"
    else:
        return "Je ne comprends pas encore cette question."

# ----------------- INTERFACE STREAMLIT -----------------

logo = Image.open("logo.png")

st.set_page_config(page_title="Build-Main AI Tools", layout="centered")
st.sidebar.title("🧠 Build-Main AI Tools")
st.sidebar.image(logo,width=200)
st.markdown("### ✍️ Entrez votre texte à résumer")
st.markdown("<hr><p style='text-align:center;'>© 2025 Build-Main | Tous droits réservés</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #5A5A5A;'>🤖 Build-Main AI Tools</h1>", unsafe_allow_html=True)
choix = st.sidebar.radio("Choisissez une fonction :", ["Résumé", "Chatbot", "Analyse de mots", "Importer un fichier texte"])

texte = ""

if choix == "Importer un fichier texte":
    fichier = st.file_uploader("Chargez un fichier .txt", type=["txt"])
    if fichier:
        texte = fichier.read().decode("utf-8")
        st.text_area("Texte importé :", texte, height=200)

elif choix == "Résumé":
    st.title("📝 Résumé automatique")
    texte = st.text_area("Entrez le texte à résumer :", height=200)
    nb = st.slider("Nombre de phrases", 1, 5, 2)
    if st.button("Générer le résumé") and texte:
        for ligne in resumer_texte(texte, nb):
            st.write("- ", ligne)

elif choix == "Chatbot":
    st.title("💬 ChatBot (local)")
    question = st.text_input("Posez une question à BuildBot :")
    if question:
        reponse = chatbot_local(question)
        st.success(reponse)

elif choix == "Analyse de mots":
    st.title("📊 Analyse des mots les plus fréquents")
    texte = st.text_area("Entrez un texte à analyser :", height=200)
    if texte:
        freqs = analyser_frequences(texte)
        mots, valeurs = zip(*freqs)
        st.bar_chart(dict(zip(mots, valeurs)))
        st.write("Top mots :", freqs)
