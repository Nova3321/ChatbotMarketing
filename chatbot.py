import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import random
import pickle
import os
import time
from datetime import datetime
import glob
import shutil
import requests
from bs4 import BeautifulSoup
import re
from urllib.robotparser import RobotFileParser
from deep_translator import GoogleTranslator
from transformers import pipeline
import os
import gdown

FILE_ID = "1MlldsoU3xCuGckVzrmS5oYKGlYcIeU-R"
FILE_URL = f"https://drive.google.com/uc?id={FILE_ID}"
FILE_PATH = "full.txt"

if not os.path.exists(FILE_PATH):
    print("Téléchargement de full.txt depuis Google Drive…")
    gdown.download(FILE_URL, FILE_PATH, quiet=False)

# Télécharger les ressources NLTK (seulement si nécessaire)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


# Cache pour les modèles avec Streamlit
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_lemmatizer():
    return WordNetLemmatizer()


model = load_model()
lemmatizer = load_lemmatizer()

# Cache file pour les embeddings avec versionnage
DATA_FILE = "full.txt"
CACHE_FILE = f"sentence_embeddings_{os.path.getmtime(DATA_FILE) if os.path.exists(DATA_FILE) else 0}.pkl"


# -------------------------------
# Prétraitement du texte amélioré
# -------------------------------
def preprocess(text):
    text = text.lower().strip()
    # Supprimer la ponctuation mais garder certains caractères importants
    text = ''.join([char for char in text if char.isalnum() or char in ' .?!,'])
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    return " ".join(tokens)


# -------------------------------
# Cache des embeddings avec vérification
# -------------------------------
def cache_embeddings(sentences, cache_file=CACHE_FILE):
    # Vérifier si le cache existe et est à jour
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            st.warning("Le fichier de cache est corrompu, recalcul des embeddings...")

    # Calculer les embeddings avec une barre de progression
    st.info("Calcul des embeddings en cours... Cette opération peut prendre quelques minutes.")
    progress_bar = st.progress(0)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)

    # Simulation de progression pour l'UI
    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)

    # Sauvegarder les embeddings
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(sentence_embeddings, f)
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde du cache: {str(e)}")

    progress_bar.empty()
    st.success("Embeddings calculés et sauvegardés avec succès!")
    return sentence_embeddings


# -------------------------------
# Réponses plus naturelles et variées
# -------------------------------
def make_response_more_human(text, query, lang='fr'):
    """
    Génère une réponse naturelle dans la langue spécifiée
    """
    if lang == 'fr':
        introductions = [
            f"Super question sur {query} !",
            f"Je vois où tu veux en venir avec '{query}' !",
            f"Intéressant, ta question sur {query} !",
        ]

        styles = [
            f"{random.choice(introductions)} Voici une explication détaillée : {text}",
            f"{random.choice(introductions)} Voici ce que je peux t'expliquer : {text}",
        ]

        follow_ups = [
            " Tu veux que je creuse un aspect particulier ?",
            " Ça répond à ta question ou tu veux plus de détails ?",
        ]
    elif lang == 'en':
        introductions = [
            f"Great question about {query}!",
            f"I see what you mean about '{query}'!",
            f"Interesting question about {query}!",
        ]

        styles = [
            f"{random.choice(introductions)} Here's a detailed explanation: {text}",
            f"{random.choice(introductions)} Here's what I can explain: {text}",
        ]

        follow_ups = [
            " Would you like me to dig into a specific aspect?",
            " Does that answer your question or would you like more details?",
        ]
    else:  # arabe
        introductions = [
            f"سؤال رائع عن {query}!",
            f"أرى ما تعنيه بـ '{query}'!",
            f"سؤال مثير للاهتمام عن {query}!",
        ]

        styles = [
            f"{random.choice(introductions)} إليك شرح مفصل: {text}",
            f"{random.choice(introductions)} إليك ما يمكنني شرحه: {text}",
        ]

        follow_ups = [
            " هل تريد أن أتعمق في جانب معين؟",
            " هل هذا يجيب على سؤالك أم تريد المزيد من التفاصيل؟",
        ]

    return random.choice(styles) + random.choice(follow_ups)


# -------------------------------
# Fonctions de réponse par défaut
# -------------------------------
def get_no_query_response(lang):
    """Retourne une réponse quand la question est vide"""
    responses = {
        "fr": "Oups, ta question semble vide ! Pose-moi une question sur le marketing, je suis prêt à t'aider !",
        "en": "Oops, your question seems empty! Ask me a question about marketing, I'm ready to help!",
        "ar": "عذرًا، يبدو أن سؤالك فارغ! اطرح عليَّ سؤالاً عن التسويق، أنا مستعد للمساعدة!"
    }
    return responses.get(lang, responses["fr"])


def get_no_results_response(query, lang):
    """Retourne une réponse quand aucun résultat n'est trouvé"""
    responses = {
        "fr": "Le marketing, c'est vaste ! Ça inclut la pub, le digital, les études de marché, et plus encore. L'objectif est de comprendre les besoins des consommateurs et de créer de la valeur. Pose-moi une question précise pour que je t'aide davantage !",
        "en": "Marketing is vast! It includes advertising, digital, market research, and more. The goal is to understand consumer needs and create value. Ask me a specific question so I can help you better!",
        "ar": "التسويق مجال واسع! يشمل الإعلان والتسويق الرقمي ودراسات السوق وغير ذلك. الهدف هو فهم احتياجات المستهلكين وخلق القيمة. اطرح عليَّ سؤالاً محددًا لكي أتمكن من مساعدتك بشكل أفضل!"
    }
    return responses.get(lang, responses["fr"])


# -------------------------------
# Trouver plusieurs phrases pertinentes avec seuil de similarité
# -------------------------------
def get_most_relevant_sentences(user_query, sentences, sentence_embeddings, top_k=3, similarity_threshold=0.5):
    """Trouve les phrases les plus pertinentes avec regroupement contextuel"""
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]

    # Obtenir les indices triés par pertinence
    sorted_indices = scores.argsort(descending=True)

    # Filtrer par seuil de similarité et regrouper par contexte
    relevant_sentences = []
    used_indices = set()

    for idx in sorted_indices:
        if scores[idx] < similarity_threshold:
            continue

        if idx in used_indices:
            continue

        # Trouver des phrases contextuellement proches
        context_group = [sentences[idx]]
        used_indices.add(idx)

        # Chercher des phrases similaires pour former un contexte cohérent
        for other_idx in sorted_indices:
            if other_idx in used_indices:
                continue

            if scores[other_idx] < similarity_threshold:
                continue

            # Vérifier la similarité avec la phrase actuelle
            similarity = util.pytorch_cos_sim(
                sentence_embeddings[idx],
                sentence_embeddings[other_idx]
            )[0].item()

            if similarity > 0.7:  # regrouper les phrases très similaires
                context_group.append(sentences[other_idx])
                used_indices.add(other_idx)

                if len(context_group) >= 3:  # Limiter la taille du groupe
                    break

        relevant_sentences.append(" ".join(context_group))

        if len(relevant_sentences) >= top_k:
            break

    if not relevant_sentences:
        return None

    # Prioriser les groupes les plus pertinents
    return " ".join(relevant_sentences[:top_k])


# -------------------------------
# Compréhension du contexte
# -------------------------------
def understand_query_context(query, conversation_history):
    """Analyse le contexte de la conversation pour mieux comprendre la requête"""
    # Si l'historique est vide, retourner la requête telle quelle
    if not conversation_history:
        return query

    # Analyser les derniers échanges pour comprendre le contexte
    last_exchanges = " ".join([msg for _, msg, _ in conversation_history[-4:] if _ == "user"])

    # Utiliser un modèle pour comprendre le contexte
    try:
        context_embedding = model.encode(last_exchanges, convert_to_tensor=True)
        query_embedding = model.encode(query, convert_to_tensor=True)

        # Calculer la similarité entre le contexte et la requête
        similarity = util.pytorch_cos_sim(context_embedding, query_embedding)[0].item()

        # Si la requête est liée au contexte, l'enrichir avec ce contexte
        if similarity > 0.4:
            enhanced_query = f"{last_exchanges} {query}"
            return enhanced_query
    except:
        pass

    return query


# -------------------------------
# Génération de réponses cohérentes
# -------------------------------
def generate_coherent_response(retrieved_text, query):
    """Transforme le texte récupéré en une réponse cohérente"""
    # Si le texte est déjà cohérent, le retourner tel quel
    sentences = nltk.sent_tokenize(retrieved_text)
    if len(sentences) <= 2:
        return retrieved_text

    # Sinon, essayer de le résumer ou de le reformuler
    try:
        # Utiliser un modèle de summarization
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
        summary = summarizer(
            retrieved_text,
            max_length=150,
            min_length=40,
            do_sample=False
        )

        return summary[0]['summary_text']
    except:
        # Fallback: prendre les premières phrases les plus pertinentes
        return ". ".join(sentences[:2]) + "."


# -------------------------------
# Traduction
# -------------------------------
def translate_text(text, target_lang):
    """Traduit le texte dans la langue choisie"""
    try:
        if target_lang == 'fr':
            return text
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        return f"[Erreur de traduction: {e}] {text}"


# -------------------------------
# Fonction chatbot principale
# -------------------------------
def chatbot(query, sentences, sentence_embeddings):
    # Récupérer la langue de réponse
    response_lang = st.session_state.get("response_language", "fr")

    if not query.strip():
        return get_no_query_response(response_lang)

    # Comprendre le contexte de la conversation
    conversation_history = st.session_state.get("history", [])
    enhanced_query = understand_query_context(query, conversation_history)

    # Rechercher les phrases pertinentes
    preprocessed_query = preprocess(enhanced_query)
    similarity_threshold = st.session_state.get("similarity_threshold", 0.5)

    best_sentences = get_most_relevant_sentences(
        preprocessed_query,
        sentences,
        sentence_embeddings,
        top_k=3,
        similarity_threshold=similarity_threshold
    )

    if not best_sentences:
        return get_no_results_response(enhanced_query, response_lang)

    # Générer une réponse cohérente
    coherent_response = generate_coherent_response(best_sentences, query)

    # Traduire si nécessaire
    if response_lang != 'fr':
        coherent_response = translate_text(coherent_response, response_lang)

    return make_response_more_human(coherent_response, query, response_lang)


# -------------------------------
# Nettoyage et organisation de la base
# -------------------------------
def clean_knowledge_base():
    """Nettoie et structure la base de connaissances"""
    try:
        with open("full.txt", "r", encoding="utf-8") as f:
            content = f.read()

        # Séparer le contenu en phrases
        sentences = nltk.sent_tokenize(content)

        # Filtrer les phrases non pertinentes
        filtered_sentences = []
        for sentence in sentences:
            # Ignorer les phrases trop courtes
            if len(sentence.split()) < 4:
                continue

            # Ignorer les phrases avec beaucoup de caractères spéciaux
            special_chars = sum(1 for c in sentence if not c.isalnum() and not c.isspace())
            if special_chars / len(sentence) > 0.3:
                continue

            # Ignorer les phrases en majuscules (souvent des titres)
            if sentence.isupper():
                continue

            filtered_sentences.append(sentence)

        # Réorganiser par thèmes
        themed_content = organize_by_themes(filtered_sentences)

        # Réécrire le fichier
        with open("full.txt", "w", encoding="utf-8") as f:
            f.write(themed_content)

        return len(sentences) - len(filtered_sentences)

    except Exception as e:
        st.error(f"Erreur lors du nettoyage: {str(e)}")
        return 0


def organize_by_themes(sentences):
    """Organise les phrases par thèmes marketing"""
    themes = {
        "STRATÉGIE MARKETING": [],
        "MARKETING DIGITAL": [],
        "RÉSEAUX SOCIAUX": [],
        "SEO": [],
        "CONTENT MARKETING": [],
        "EMAIL MARKETING": [],
        "ANALYSE ET MÉTRIQUES": [],
        "ÉTUDES DE CAS": []
    }

    # Mots-clés par thème
    keywords = {
        "STRATÉGIE MARKETING": ["stratégie", "plan", "positionnement", "ciblage", "segmentation", "mix", "4P"],
        "MARKETING DIGITAL": ["digital", "en ligne", "online", "web", "internet", "numérique"],
        "RÉSEAUX SOCIAUX": ["facebook", "instagram", "twitter", "linkedin", "social", "réseaux sociaux", "community"],
        "SEO": ["seo", "référencement", "moteur de recherche", "google", "ranking", "backlink", "mot-clé"],
        "CONTENT MARKETING": ["contenu", "content", "blog", "article", "rédaction", "storytelling"],
        "EMAIL MARKETING": ["email", "newsletter", "campagne email", "mailing", "automatisation"],
        "ANALYSE ET MÉTRIQUES": ["analytique", "métrique", "kpi", "roi", "mesure", "performance", "conversion"],
        "ÉTUDES DE CAS": ["cas", "exemple", "étude", "success story", "best practice"]
    }

    # Classer les phrases par thème
    for sentence in sentences:
        sentence_lower = sentence.lower()
        assigned = False

        for theme, keys in keywords.items():
            if any(key in sentence_lower for key in keys):
                themes[theme].append(sentence)
                assigned = True
                break

        if not assigned:
            # Par défaut, mettre dans stratégie marketing
            themes["STRATÉGIE MARKETING"].append(sentence)

    # Construire le contenu organisé
    organized_content = []
    for theme, sentences_in_theme in themes.items():
        if sentences_in_theme:
            organized_content.append(f"# {theme}")
            organized_content.extend(sentences_in_theme)
            organized_content.append("")  # Ligne vide entre les thèmes

    return "\n".join(organized_content)


# -------------------------------
# Charger et préparer les données
# -------------------------------
def load_data():
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw_text = f.read()

        if not raw_text.strip():
            st.error("Le fichier de données est vide.")
            return [], []

        sentences = nltk.sent_tokenize(raw_text)
        if not sentences:
            st.error("Aucune phrase n'a pu être extraite du fichier.")
            return [], []

        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = [preprocess(s) for s in sentences]
        sentence_embeddings = cache_embeddings(sentences)

        return sentences, sentence_embeddings

    except FileNotFoundError:
        st.error("Erreur : Le fichier 'full.txt' est introuvable. Veuillez vérifier son emplacement.")
        return [], []
    except UnicodeDecodeError:
        st.error("Erreur : Problème d'encodage du fichier. Assurez-vous qu'il est en UTF-8.")
        return [], []
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement des données : {str(e)}")
        return [], []


# -------------------------------
# Import de fichiers
# -------------------------------
def merge_text_files():
    """Fusionne tous les fichiers texte du dossier 'sources' dans full.txt"""
    sources_dir = "sources"
    os.makedirs(sources_dir, exist_ok=True)

    # Trouver tous les fichiers texte dans le dossier sources
    text_files = glob.glob(os.path.join(sources_dir, "*.txt"))

    if not text_files:
        return False

    # Lire le contenu de tous les fichiers
    all_content = []
    for file_path in text_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    filename = os.path.basename(file_path)
                    all_content.append(
                        f"\n\n# CONTENU DE {filename} (IMPORTÉ LE {datetime.now().strftime('%Y-%m-%d')})\n{content}")
        except Exception as e:
            st.error(f"Erreur avec le fichier {file_path}: {str(e)}")

    if all_content:
        # Ajouter le contenu à full.txt
        with open("full.txt", "a", encoding="utf-8") as f:
            f.write("\n".join(all_content))

        # Déplacer les fichiers importés vers un dossier "imported" pour éviter les doublons
        imported_dir = os.path.join(sources_dir, "imported")
        os.makedirs(imported_dir, exist_ok=True)

        for file_path in text_files:
            filename = os.path.basename(file_path)
            shutil.move(file_path, os.path.join(imported_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"))

        return True

    return False


def file_import_interface():
    """Interface pour importer des fichiers"""
    st.sidebar.header("📤 Importer des fichiers")

    uploaded_files = st.sidebar.file_uploader(
        "Choisissez un ou plusieurs fichiers texte",
        type=['txt'],
        help="Les fichiers seront ajoutés à la base de connaissances",
        accept_multiple_files=True
    )

    if uploaded_files:
        sources_dir = "sources"
        os.makedirs(sources_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            # Sauvegarder le fichier dans le dossier sources
            with open(os.path.join(sources_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.sidebar.success(f"{len(uploaded_files)} fichier(s) téléchargé(s) dans le dossier 'sources'")

    if st.sidebar.button("Fusionner les fichiers dans la base"):
        with st.spinner("Fusion des fichiers en cours..."):
            if merge_text_files():
                st.sidebar.success("Fichiers fusionnés avec succès!")

                # Supprimer le cache des embeddings
                cache_files = [f for f in os.listdir(".") if f.startswith("sentence_embeddings")]
                for cache_file in cache_files:
                    os.remove(cache_file)

                st.rerun()
            else:
                st.sidebar.warning("Aucun fichier à fusionner ou erreur lors de la fusion.")


# -------------------------------
# Scraping de contenu
# -------------------------------
def check_robots_txt(url):
    """Vérifie si le scraping est autorisé par robots.txt"""
    try:
        base_url = "/".join(url.split("/")[:3])
        rp = RobotFileParser()
        rp.set_url(f"{base_url}/robots.txt")
        rp.read()
        return rp.can_fetch("*", url)
    except:
        return True  # Si on ne peut pas vérifier, on continue avec prudence


def scrape_marketing_blog(url, depth=1):
    """Scrape le contenu d'un blog marketing"""
    try:
        # Vérifier robots.txt
        if not check_robots_txt(url):
            st.warning(f"Scraping non autorisé pour {url} selon robots.txt")
            return None

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Supprimer les éléments indésirables
        for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'form']):
            element.decompose()

        # Extraire le titre
        title = soup.find('h1')
        title_text = title.get_text().strip() if title else "Sans titre"

        # Extraire le contenu principal - stratégies pour différents sites
        content = ""

        # Stratégies courantes pour les blogs marketing
        selectors = [
            'article',
            '.post-content',
            '.entry-content',
            '.article-content',
            '.blog-post',
            '[class*="content"]',
            'main'
        ]

        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    text = element.get_text(separator=' ', strip=True)
                    if len(text.split()) > 50:  # Un contenu significatif
                        content = text
                        break
                if content:
                    break

        # Si on n'a pas trouvé de contenu avec les sélecteurs, on prend tout le body
        if not content:
            body = soup.find('body')
            if body:
                content = body.get_text(separator=' ', strip=True)

        # Nettoyer le contenu
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'[^\w\s.,;:!?()-]', '', content)

        # Limiter la longueur
        if len(content) > 10000:
            content = content[:10000] + "..."

        return f"{title_text}. {content}"

    except Exception as e:
        st.error(f"Erreur lors du scraping de {url}: {str(e)}")
        return None


def scrape_multiple_urls(urls):
    """Scrape plusieurs URLs et retourne le contenu combiné"""
    all_content = []

    for url in urls:
        if not url.startswith('http'):
            url = 'https://' + url

        st.info(f"Scraping de {url}...")
        content = scrape_marketing_blog(url)

        if content:
            all_content.append(f"\n\n# CONTENU DE {url} (SCRAPÉ LE {datetime.now().strftime('%Y-%m-%d')})\n{content}")

        # Pause pour éviter de surcharger le serveur
        time.sleep(2)

    return "\n".join(all_content)


def scraping_interface():
    """Interface pour le scraping de blogs"""
    st.sidebar.header("🌐 Scraping de Blogs Marketing")

    st.sidebar.info("Entrez les URLs des blogs marketing à scraper (une par ligne)")

    urls_text = st.sidebar.text_area(
        "URLs à scraper:",
        height=100,
        help="Ex: blog.hubspot.com/marketing\nneilpatel.com/blog\ncontentmarketinginstitute.com"
    )

    if st.sidebar.button("Lancer le scraping"):
        if not urls_text.strip():
            st.sidebar.warning("Veuillez entrer au moins une URL.")
            return

        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]

        with st.spinner(f"Scraping de {len(urls)} URL(s) en cours..."):
            scraped_content = scrape_multiple_urls(urls)

            if scraped_content:
                # Ajouter le contenu à la base
                with open("full.txt", "a", encoding="utf-8") as f:
                    f.write(scraped_content)

                st.sidebar.success(f"Scraping terminé! {len(scraped_content.split())} mots ajoutés.")

                # Supprimer le cache
                cache_files = [f for f in os.listdir(".") if f.startswith("sentence_embeddings")]
                for cache_file in cache_files:
                    os.remove(cache_file)

                st.rerun()
            else:
                st.sidebar.error("Aucun contenu n'a pu être scrapé.")


# -------------------------------
# Gestion de la qualité
# -------------------------------
def remove_duplicates():
    """Supprime les doublons approximatifs du fichier full.txt"""
    try:
        with open("full.txt", "r", encoding="utf-8") as f:
            content = f.read()

        # Séparer le contenu en sections
        sections = re.split(r'#.*?\n', content)
        sections = [s.strip() for s in sections if s.strip()]

        # Encoder les sections pour comparer leur similarité
        if len(sections) > 1:
            embeddings = model.encode(sections, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(embeddings, embeddings)

            # Identifier les doublons
            to_remove = set()
            for i in range(len(similarities)):
                for j in range(i + 1, len(similarities)):
                    if similarities[i][j] > 0.95:  # Seuil de similarité élevé
                        to_remove.add(j)

            # Recréer le contenu sans les doublons
            new_content = []
            for i, section in enumerate(sections):
                if i not in to_remove:
                    new_content.append(section)

            # Réécrire le fichier
            with open("full.txt", "w", encoding="utf-8") as f:
                f.write("\n\n".join(new_content))

            return len(to_remove)

        return 0

    except Exception as e:
        st.error(f"Erreur lors de la suppression des doublons: {str(e)}")
        return 0


def quality_check():
    """Vérifie la qualité de la base de données"""
    try:
        with open("full.txt", "r", encoding="utf-8") as f:
            content = f.read()

        sentences = nltk.sent_tokenize(content)
        words = content.split()

        # Calculer la diversité lexicale
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words) if words else 0

        # Vérifier la longueur moyenne des phrases
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        return {
            "sentences": len(sentences),
            "words": len(words),
            "unique_words": len(unique_words),
            "diversity_ratio": diversity_ratio,
            "avg_sentence_length": avg_sentence_length
        }

    except Exception as e:
        st.error(f"Erreur lors de l'analyse de qualité: {str(e)}")
        return None


def quality_management_interface():
    """Interface pour gérer la qualité des réponses"""
    st.sidebar.header("🛠️ Amélioration de la Qualité")

    if st.sidebar.button("Nettoyer la base de connaissances"):
        removed_count = clean_knowledge_base()
        st.sidebar.success(f"{removed_count} phrases non pertinentes supprimées!")

        # Recréer les embeddings
        global sentences, sentence_embeddings
        sentences, sentence_embeddings = load_data()

        st.rerun()

    if st.sidebar.button("Réorganiser par thèmes"):
        with st.spinner("Réorganisation en cours..."):
            organize_by_themes(sentences)
            st.sidebar.success("Base réorganisée par thèmes!")

            # Recréer les embeddings
            sentences, sentence_embeddings = load_data()
            st.rerun()

    # Paramètres de qualité
    st.sidebar.subheader("Paramètres de pertinence")
    similarity_threshold = st.sidebar.slider(
        "Seuil de similarité",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Augmentez cette valeur pour des réponses plus précises mais moins nombreuses"
    )

    st.session_state.similarity_threshold = similarity_threshold


# -------------------------------
# Interface Streamlit améliorée
# -------------------------------
def main():
    st.set_page_config(
        page_title="Chatbot Marketing",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS personnalisé
    st.markdown("""
    <style>
    .stChatInput input {
        background-color: #f0f2f6;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e6f7ff;
    }
    .bot-message {
        background-color: #f9f9f9;
    }
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialisation de l'historique
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Initialisation de la langue
    if "response_language" not in st.session_state:
        st.session_state["response_language"] = "fr"

    # Initialisation du seuil de similarité
    if "similarity_threshold" not in st.session_state:
        st.session_state["similarity_threshold"] = 0.5

    # Sidebar avec informations supplémentaires
    with st.sidebar:
        st.title("À propos")
        st.info("""
        Ce chatbot utilise l'IA pour répondre à vos questions sur le marketing.
        Il analyse votre question et trouve les informations les plus pertinentes
        dans sa base de connaissances.
        """)

        # Sélecteur de langue
        st.header("🌍 Langue de Réponse")
        language_option = st.radio(
            "Choisissez la langue de réponse:",
            ["Français", "English", "العربية"],
            index=0
        )

        # Mapper les options aux codes de langue
        language_map = {
            "Français": "fr",
            "English": "en",
            "العربية": "ar"
        }

        # Stocker la langue sélectionnée dans session_state
        st.session_state.response_language = language_map[language_option]

        if st.button("Effacer l'historique de conversation"):
            st.session_state["history"] = []
            st.rerun()

        st.divider()

        # Enrichissement de la base
        st.header("📚 Enrichir la Base de Données")
        file_import_interface()
        st.divider()
        scraping_interface()
        st.divider()

        # Gestion de la qualité
        quality_management_interface()
        st.divider()

        # Outils d'optimisation
        st.header("🔧 Outils d'Optimisation")
        if st.button("Supprimer les doublons"):
            removed_count = remove_duplicates()
            if removed_count > 0:
                st.success(f"{removed_count} doublon(s) supprimé(s)!")
                # Supprimer le cache
                cache_files = [f for f in os.listdir(".") if f.startswith("sentence_embeddings")]
                for cache_file in cache_files:
                    os.remove(cache_file)
                st.rerun()
            else:
                st.info("Aucun doublon détecté.")

        if st.button("Vérifier la qualité"):
            quality_stats = quality_check()
            if quality_stats:
                st.info(
                    f"📊 Qualité de la base:\n"
                    f"- Phrases: {quality_stats['sentences']}\n"
                    f"- Mots: {quality_stats['words']}\n"
                    f"- Mots uniques: {quality_stats['unique_words']}\n"
                    f"- Diversité lexicale: {quality_stats['diversity_ratio']:.2%}\n"
                    f"- Longueur moyenne des phrases: {quality_stats['avg_sentence_length']:.1f} mots"
                )

        # Statistiques de la base
        try:
            with open("full.txt", "r", encoding="utf-8") as f:
                content = f.read()
            sentences_count = len(nltk.sent_tokenize(content))
            word_count = len(content.split())
            st.divider()
            st.info(f"📊 Taille de la base:\n- {sentences_count} phrases\n- {word_count} mots")

            # Afficher la date de la dernière modification
            mtime = os.path.getmtime("full.txt")
            st.caption(f"Dernière modification: {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

        st.divider()
        st.caption("Développé avec Streamlit et Sentence Transformers")

    # Colonne principale
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("💬 Chatbot Marketing Moderne")
        st.markdown("Posez vos questions sur le marketing, je vous réponds de manière claire et détaillée ! 🚀")

        # Charger les données avec indicateur de progression
        data_load_state = st.info("Chargement des données en cours...")
        sentences, sentence_embeddings = load_data()
        data_load_state.empty()

        if not sentences:
            st.warning("Impossible de charger les données. Vérifiez le fichier 'full.txt' et réessayez.")
            return

        # Suggestions de questions
        question_suggestions = [
            "Qu'est-ce que le marketing digital ?",
            "Comment créer une stratégie marketing ?",
            "Quelles sont les tendances actuelles en marketing ?",
            "Comment mesurer l'efficacité d'une campagne marketing ?"
        ]

        st.write("**Suggestions de questions :**")
        cols = st.columns(2)
        for i, question in enumerate(question_suggestions):
            with cols[i % 2]:
                if st.button(question, key=f"suggestion_{i}"):
                    st.session_state.chat_input = question

        # Zone de saisie style chat
        user_query = st.chat_input("Écrivez votre question ici...", key="chat_input")

        if user_query:
            with st.spinner("Je réfléchis à une réponse..."):
                response = chatbot(user_query, sentences, sentence_embeddings)
                timestamp = datetime.now().strftime("%H:%M")
                st.session_state["history"].append(("user", user_query, timestamp))
                st.session_state["history"].append(("assistant", response, timestamp))

        # Affichage de l'historique sous forme de chat
        for sender, msg, time in st.session_state["history"]:
            if sender == "user":
                with st.chat_message("user"):
                    # Afficher la langue détectée pour la question
                    lang = st.session_state.get("response_language", "fr")
                    lang_label = "🇫🇷" if lang == "fr" else "🇬🇧" if lang == "en" else "🇸🇦"
                    st.markdown(f"**Vous** {lang_label} ({time}): {msg}")
            else:
                with st.chat_message("assistant"):
                    # Afficher la langue de la réponse
                    response_lang = st.session_state.get("response_language", "fr")
                    lang_label = "🇫🇷" if response_lang == "fr" else "🇬🇧" if response_lang == "en" else "🇸🇦"

                    # Appliquer le style RTL pour l'arabe
                    if response_lang == "ar":
                        st.markdown(f'<div class="rtl-text">**المساعد** {lang_label} ({time}): {msg}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Assistant** {lang_label} ({time}): {msg}")

    with col2:
        st.write("**📊 Statistiques**")
        st.metric("Messages échangés", len([h for h in st.session_state["history"] if h[0] == "user"]))
        st.metric("Phrases dans la base", len(sentences))

        if st.session_state["history"]:
            st.divider()
            st.write("**📝 Historique récent**")
            for sender, msg, time in st.session_state["history"][-6:]:
                if sender == "user":
                    st.caption(f"Vous ({time}): {msg[:50]}..." if len(msg) > 50 else f"Vous ({time}): {msg}")


if __name__ == "__main__":
    main()