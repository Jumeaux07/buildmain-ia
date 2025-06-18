import streamlit as st
import nltk
import heapq
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import csv
import os

# ----------------- CONFIGURATION NLTK -----------------
def safe_nltk_download(resource):
    """Télécharge les ressources NLTK de manière sécurisée"""
    try:
        nltk.data.find(resource)
    except LookupError:
        try:
            # Essaie d'abord avec le nom complet
            nltk.download(resource.split("/")[-1])
        except:
            # Si ça échoue, essaie avec des alternatives
            if "punkt" in resource:
                try:
                    nltk.download("punkt_tab")
                except:
                    nltk.download("punkt")
            elif "stopwords" in resource:
                nltk.download("stopwords")

# Télécharger les ressources nécessaires avec gestion d'erreur améliorée
@st.cache_resource
def init_nltk():
    """Initialise NLTK avec cache pour éviter les téléchargements répétés"""
    try:
        safe_nltk_download("tokenizers/punkt")
        safe_nltk_download("tokenizers/punkt_tab")
        safe_nltk_download("corpora/stopwords")
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des ressources NLTK: {e}")
        # Téléchargement direct en cas d'échec
        try:
            nltk.download("punkt_tab", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
        except:
            pass

# Initialiser NLTK
init_nltk()

# ----------------- FONCTIONS -----------------
def enregistrer_texte_utilise(action, texte, resultat=""):
    """Enregistre l'historique des actions dans un fichier CSV"""
    try:
        now = datetime.now().strftime("%d-%m-%Y %H:%M")
        with open("historique.csv", mode='a', newline='', encoding='utf-8') as historique:
            writer = csv.writer(historique)
            writer.writerow([now, action, texte[:100], resultat[:100]])
    except Exception as e:
        st.warning(f"Impossible d'enregistrer l'historique: {e}")

def resumer_texte(texte, nb_phrases=2):
    """Résume un texte en sélectionnant les phrases les plus importantes"""
    try:
        # Tokenisation des phrases avec gestion d'erreur
        try:
            phrases = sent_tokenize(texte, language='french')
        except:
            # Fallback: utilise un tokenizer plus simple si NLTK échoue
            phrases = texte.split('.')
            phrases = [p.strip() + '.' for p in phrases if p.strip()]
        
        if len(phrases) <= nb_phrases:
            return phrases
        
        # Tokenisation des mots
        try:
            mots = word_tokenize(texte.lower())
        except:
            # Fallback simple
            mots = texte.lower().split()
        
        # Récupération des stopwords
        try:
            stop_words = set(stopwords.words("french"))
        except:
            # Fallback avec des stopwords basiques en français
            stop_words = set(['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus', 'par', 'grand', 'en', 'on', 'aller', 'pouvoir', 'aussi', 'du', 'au', 'si', 'te', 'mais', 'ou', 'bien', 'non', 'comme', 'faire', 'sa', 'voir', 'lui', 'nous', 'je', 'me', 'ma', 'très'])
        
        # Filtrage des mots utiles
        mots_utiles = [mot for mot in mots if mot.isalnum() and mot not in stop_words and len(mot) > 2]
        
        if not mots_utiles:
            return phrases[:nb_phrases]
        
        # Calcul des fréquences
        frequences = Counter(mots_utiles)
        max_freq = max(frequences.values())
        
        # Normalisation des fréquences
        for mot in frequences:
            frequences[mot] /= max_freq

        # Calcul du score de chaque phrase
        scores_phrases = {}
        for phrase in phrases:
            score = 0
            try:
                mots_phrase = word_tokenize(phrase.lower())
            except:
                mots_phrase = phrase.lower().split()
            
            for mot in mots_phrase:
                if mot in frequences and len(phrase.split(" ")) < 30:
                    score += frequences[mot]
            
            if score > 0:
                scores_phrases[phrase] = score
        
        if not scores_phrases:
            return phrases[:nb_phrases]
        
        # Retour des meilleures phrases
        return heapq.nlargest(nb_phrases, scores_phrases, key=scores_phrases.get)
    
    except Exception as e:
        st.error(f"Erreur lors du résumé: {e}")
        return texte.split('.')[:nb_phrases]

def analyser_frequences(texte):
    """Analyse la fréquence des mots dans un texte"""
    try:
        try:
            mots = word_tokenize(texte.lower())
        except:
            mots = texte.lower().split()
        
        try:
            stop_words = set(stopwords.words("french"))
        except:
            stop_words = set(['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus', 'par', 'grand', 'en', 'on', 'aller', 'pouvoir', 'aussi', 'du', 'au', 'si', 'te', 'mais', 'ou', 'bien', 'non', 'comme', 'faire', 'sa', 'voir', 'lui', 'nous', 'je', 'me', 'ma', 'très'])
        
        mots_utiles = [mot for mot in mots if mot.isalnum() and mot not in stop_words and len(mot) > 2]
        return Counter(mots_utiles).most_common(10)
    except Exception as e:
        st.error(f"Erreur lors de l'analyse: {e}")
        return []

def chatbot_local(message):
    """Chatbot simple avec réponses prédéfinies"""
    message = message.lower()
    if "bonjour" in message or "salut" in message:
        return "Bonjour ! Je suis BuildBot. En quoi puis-je vous aider ?"
    elif "aide" in message or "quoi" in message or "comment" in message:
        return "Je peux résumer du texte, analyser les mots fréquents ou répondre à des questions simples. Utilisez le menu à gauche pour choisir une fonction."
    elif "merci" in message:
        return "Avec plaisir ! 😊"
    elif "au revoir" in message or "bye" in message:
        return "Au revoir ! À bientôt ! 👋"
    elif "qui es-tu" in message or "qui êtes-vous" in message:
        return "Je suis BuildBot, un assistant IA pour analyser et résumer du texte."
    else:
        return "Je ne comprends pas encore cette question. Essayez de me demander de l'aide ou utilisez les fonctions du menu."

# ----------------- INTERFACE STREAMLIT -----------------
def main():
    # Configuration de la page
    st.set_page_config(
        page_title="Build-Main AI Tools", 
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    st.sidebar.title("🧠 Build-Main AI Tools")
    
    # Tentative de chargement du logo
    try:
        logo = Image.open("logo.png")
        st.sidebar.image(logo, width=200)
    except:
        st.sidebar.write("🤖 Logo non trouvé")

    # Titre principal
    st.markdown(
        "<h1 style='text-align: center; color: #5A5A5A;'>🤖 Build-Main AI Tools</h1>", 
        unsafe_allow_html=True
    )
    
    # Menu de choix
    choix = st.sidebar.radio(
        "Choisissez une fonction :", 
        ["Résumé", "Chatbot", "Analyse de mots", "Importer un fichier texte"]
    )

    texte = ""

    # Interface selon le choix
    if choix == "Importer un fichier texte":
        st.title("📁 Importation de fichier")
        fichier = st.file_uploader("Chargez un fichier .txt", type=["txt"])
        if fichier:
            try:
                texte = fichier.read().decode("utf-8")
                st.success("Fichier chargé avec succès !")
                st.text_area("Texte importé :", texte, height=200)
                
                # Options supplémentaires après import
                st.subheader("Actions disponibles:")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Résumer ce texte"):
                        st.session_state.texte_importe = texte
                        st.session_state.page_active = "Résumé"
                        st.rerun()
                
                with col2:
                    if st.button("Analyser les mots"):
                        st.session_state.texte_importe = texte
                        st.session_state.page_active = "Analyse de mots"
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier: {e}")

    elif choix == "Résumé":
        st.title("📝 Résumé automatique")
        
        # Vérifier s'il y a du texte importé
        if hasattr(st.session_state, 'texte_importe') and hasattr(st.session_state, 'page_active') and st.session_state.page_active == "Résumé":
            texte = st.text_area("Texte à résumer :", st.session_state.texte_importe, height=200)
            # Nettoyer la session
            del st.session_state.texte_importe
            del st.session_state.page_active
        else:
            texte = st.text_area("Entrez le texte à résumer :", height=200)
        
        nb = st.slider("Nombre de phrases dans le résumé", 1, 5, 2)
        
        if st.button("Générer le résumé") and texte.strip():
            with st.spinner("Génération du résumé..."):
                lignes = resumer_texte(texte, nb)
                
                if lignes:
                    st.subheader("📋 Résumé généré:")
                    for i, ligne in enumerate(lignes, 1):
                        st.write(f"{i}. {ligne}")
                    
                    # Enregistrement
                    enregistrer_texte_utilise("Résumé", texte, " ".join(lignes))
                    st.success("Résumé généré avec succès !")
                else:
                    st.error("Impossible de générer un résumé pour ce texte.")
        elif st.button("Générer le résumé") and not texte.strip():
            st.warning("Veuillez saisir du texte à résumer.")

    elif choix == "Chatbot":
        st.title("💬 ChatBot (local)")
        
        # Historique des conversations
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Affichage de l'historique
        for message in st.session_state.chat_history:
            if message['type'] == 'user':
                st.write(f"👤 **Vous:** {message['content']}")
            else:
                st.write(f"🤖 **BuildBot:** {message['content']}")
        
        # Input pour nouvelle question
        question = st.text_input("Posez une question à BuildBot :", key="chat_input")
        
        if st.button("Envoyer") and question.strip():
            # Ajouter la question à l'historique
            st.session_state.chat_history.append({'type': 'user', 'content': question})
            
            # Générer et ajouter la réponse
            reponse = chatbot_local(question)
            st.session_state.chat_history.append({'type': 'bot', 'content': reponse})
            
            # Rafraîchir la page pour afficher la conversation
            st.rerun()
        
        # Bouton pour effacer l'historique
        if st.button("Effacer l'historique"):
            st.session_state.chat_history = []
            st.rerun()

    elif choix == "Analyse de mots":
        st.title("📊 Analyse des mots les plus fréquents")
        
        # Vérifier s'il y a du texte importé
        if hasattr(st.session_state, 'texte_importe') and hasattr(st.session_state, 'page_active') and st.session_state.page_active == "Analyse de mots":
            texte = st.text_area("Texte à analyser :", st.session_state.texte_importe, height=200)
            # Nettoyer la session
            del st.session_state.texte_importe
            del st.session_state.page_active
        else:
            texte = st.text_area("Entrez un texte à analyser :", height=200)
        
        if texte.strip():
            with st.spinner("Analyse en cours..."):
                freqs = analyser_frequences(texte)
                
                if freqs:
                    st.subheader("📈 Graphique des mots les plus fréquents")
                    mots, valeurs = zip(*freqs)
                    
                    # Créer un dictionnaire pour le graphique
                    data_chart = {mot: val for mot, val in zip(mots, valeurs)}
                    st.bar_chart(data_chart)
                    
                    st.subheader("📝 Top 10 des mots les plus fréquents")
                    for i, (mot, freq) in enumerate(freqs, 1):
                        st.write(f"{i}. **{mot}** : {freq} occurrences")
                    
                    # Statistiques supplémentaires
                    st.subheader("📊 Statistiques du texte")
                    nb_mots_total = len(texte.split())
                    nb_mots_uniques = len(set(texte.lower().split()))
                    nb_caracteres = len(texte)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mots total", nb_mots_total)
                    with col2:
                        st.metric("Mots uniques", nb_mots_uniques)
                    with col3:
                        st.metric("Caractères", nb_caracteres)
                    
                    enregistrer_texte_utilise("Analyse", texte, str(freqs[:3]))
                else:
                    st.warning("Aucun mot significatif trouvé dans le texte.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Build-Main AI Tools** v2.0")
    st.sidebar.markdown("Développé avec ❤️ et Streamlit")

if __name__ == "__main__":
    main()