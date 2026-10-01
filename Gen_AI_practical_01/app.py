import json
import subprocess
import sys
import time
from collections import Counter

import nltk
import pandas as pd
import spacy
import streamlit as st

try:
    import plotly.express as px
except ImportError:
    px = None

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NLP Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================
def apply_custom_css():
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --bg: #001f2e;
            --bg2: #002836;
            --sidebar: #00364a;
            --card: #003d4d;
            --card2: #002f3f;
            --primary: #00c996;
            --secondary: #00a88f;
            --accent: #00e6b0;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
            --main-heading: #FFFFFF;
            --body-text: #E0F5F0;
            --label: #B8E8E0;
            --secondary-text: #7FC9BC;
            --muted: #4A8A7F;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 20% 80%, rgba(0,201,150,.12) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(0,168,143,.1) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(0,230,176,.08) 0%, transparent 60%),
                linear-gradient(135deg, #001f2e 0%, #00364a 25%, #002836 50%, #003d4d 75%, #001f2e 100%);
            color: var(--body-text) !important;
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
            animation: bgShift 15s ease-in-out infinite;
        }

        @keyframes bgShift {
            0%, 100% {
                background:
                    radial-gradient(circle at 20% 80%, rgba(0,201,150,.12) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(0,168,143,.1) 0%, transparent 50%),
                    radial-gradient(circle at 50% 50%, rgba(0,230,176,.08) 0%, transparent 60%),
                    linear-gradient(135deg, #001f2e 0%, #00364a 25%, #002836 50%, #003d4d 75%, #001f2e 100%);
            }
            50% {
                background:
                    radial-gradient(circle at 80% 20%, rgba(0,201,150,.12) 0%, transparent 50%),
                    radial-gradient(circle at 20% 80%, rgba(0,168,143,.1) 0%, transparent 50%),
                    radial-gradient(circle at 50% 50%, rgba(0,230,176,.1) 0%, transparent 60%),
                    linear-gradient(135deg, #002836 0%, #003d4d 25%, #001f2e 50%, #00364a 75%, #002836 100%);
            }
        }

        [data-testid="stHeader"] {
            background: rgba(10,14,39,.92) !important;
            backdrop-filter: blur(12px) !important;
        }

        /* ---------------- SIDEBAR ---------------- */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #00364a 0%, #002836 100%) !important;
            border-right: 1px solid rgba(0,201,150,.15);
        }

        [data-testid="stSidebar"] * {
            color: var(--body-text) !important;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--main-heading) !important;
        }

        /* Make radio navigation text clearly visible */
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            color: var(--label) !important;
            opacity: 1 !important;
            transition: all .2s ease !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label div {
            color: var(--label) !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover div {
            color: var(--main-heading) !important;
        }

        /* Radio circles */
        [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
            gap: 8px;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding: 10px 12px !important;
            border-radius: 12px !important;
            transition: all .22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(0,201,150,.15) !important;
            transform: translateX(4px);
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: linear-gradient(
                90deg,
                rgba(0,201,150,.25),
                rgba(0,168,143,.15)
            ) !important;
            border: 1px solid rgba(0,201,150,.35);
            box-shadow: 0 4px 12px rgba(0,201,150,.2);
        }

        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--secondary-text) !important;
        }

        /* Divider styling */
        [data-testid="stSidebar"] [data-testid="stDivider"] {
            border-color: rgba(0,201,150,.15) !important;
        }

        /* ---------------- HEADER ---------------- */
        .top-header {
            position: sticky;
            top: 0;
            z-index: 50;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            padding: 14px 18px;
            margin-bottom: 18px;
            background: rgba(0,61,77,.5);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(0,201,150,.15);
            border-radius: 18px;
            box-shadow: 0 8px 32px rgba(0,0,0,.2);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-icon {
            width: 44px;
            height: 44px;
            display: grid;
            place-items: center;
            border-radius: 14px;
            background: linear-gradient(135deg, #00c996, #00a88f);
            font-size: 22px;
            box-shadow: 0 4px 16px rgba(0,201,150,.3);
        }

        .brand-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--main-heading);
            letter-spacing: -0.3px;
        }

        .brand-subtitle {
            font-size: .8rem;
            color: var(--secondary-text);
            margin-top: 2px;
        }

        /* ---------------- CARDS ---------------- */
        .glass {
            background: linear-gradient(
                180deg,
                rgba(0,61,77,.5),
                rgba(0,40,54,.4)
            );
            border: 1px solid rgba(0,201,150,.15);
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,.15);
            transition: all .3s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(12px);
        }

        .glass:hover {
            transform: translateY(-4px);
            border-color: rgba(0,201,150,.25);
            box-shadow: 0 16px 48px rgba(0,201,150,.12);
        }

        .stat {
            background: linear-gradient(
                135deg,
                rgba(0,201,150,.08),
                rgba(0,168,143,.05)
            ) !important;
            margin: 10px !important;
        }

        .stat .label {
            font-size: .8rem;
            color: var(--secondary-text) !important;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stat .value {
            font-size: 1.75rem;
            font-weight: 900;
            background: linear-gradient(135deg, #00e6b0, #00c996);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-top: 6px;
        }

        .subtle {
            font-size: .75rem;
            color: var(--muted) !important;
        }

        .section-title {
            color: var(--main-heading);
            font-size: 1.4rem;
            font-weight: 800;
            margin: 12px 0 16px;
            letter-spacing: -0.5px;
        }

        /* ---------------- INPUTS ---------------- */
        textarea,
        input {
            color: var(--body-text) !important;
        }

        .stTextArea textarea {
            color: var(--body-text) !important;
            background: rgba(0,40,54,.6) !important;
            border: 1.5px solid rgba(0,201,150,.15) !important;
            border-radius: 14px !important;
            transition: all .2s ease !important;
        }

        .stTextArea textarea:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 2px rgba(0,201,150,.2) !important;
            background: rgba(0,40,54,.9) !important;
        }

        .stTextArea textarea::placeholder {
            color: var(--muted) !important;
        }

        [data-testid="stTextInput"] input {
            background: rgba(224,245,240,.95) !important;
            color: #001f2e !important;
            border: 1.5px solid rgba(0,201,150,.2) !important;
            border-radius: 12px !important;
            transition: all .2s ease !important;
        }

        [data-testid="stTextInput"] input:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 2px rgba(0,201,150,.2) !important;
        }

        [data-testid="stTextInput"] input::placeholder {
            color: var(--muted) !important;
        }

        /* ---------------- BUTTONS ---------------- */
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 12px !important;
            border: 1.5px solid rgba(0,201,150,.3) !important;
            background: linear-gradient(135deg, #00c996, #008872) !important;
            color: var(--main-heading) !important;
            font-weight: 700 !important;
            min-height: 44px;
            transition: all .25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 12px rgba(0,201,150,.25) !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: #00e6b0 !important;
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,201,150,.35) !important;
        }

        .stButton > button:active,
        .stDownloadButton > button:active {
            transform: translateY(0);
        }

        /* Sidebar uploader button */
        [data-testid="stSidebar"] .stFileUploader button {
            background: linear-gradient(135deg, #00c996, #008872) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            transition: all .2s ease !important;
        }

        [data-testid="stSidebar"] .stFileUploader button:hover {
            box-shadow: 0 4px 12px rgba(0,201,150,.35) !important;
        }

        /* ---------------- DATAFRAME ---------------- */
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(0,201,150,.15);
            border-radius: 14px;
            overflow: hidden;
            background: rgba(0,40,54,.3);
        }

        [data-testid="stDataFrame"] > div {
            background: rgba(0,40,54,.3) !important;
        }

        /* ---------------- ANIMATION ---------------- */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(12px);
            }
            to {
                opacity: 1;
                transform: none;
            }
        }

        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: none;
            }
        }

        .fade {
            animation: fadeInUp .5s cubic-bezier(0.4, 0, 0.2, 1) both;
        }

        /* Smooth scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(0,201,150,.05);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(0,201,150,.3);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(0,201,150,.5);
        }

        /* ---------------- MOBILE ---------------- */
        @media (max-width: 800px) {
            .top-header {
                flex-direction: column;
                align-items: stretch;
            }
            
            .section-title {
                font-size: 1.25rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HELPERS
# ============================================================
def render_metric_card(title, value, subtitle, icon="📈"):
    st.markdown(
        f"""
        <div class="glass stat fade">
            <div style="display:flex;align-items:center;gap:.8rem">
                <div style="font-size:22px">{icon}</div>
                <div>
                    <div class="label">{title}</div>
                    <div class="value">{value}</div>
                    <div class="subtle">{subtitle}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def ensure_nltk_data():
    packages = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "omw-1.4",
    ]

    for package in packages:
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass


@st.cache_resource
def load_spacy_model(model_name="en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True,
                capture_output=True,
                text=True,
            )
            return spacy.load(model_name)
        except Exception:
            # Blank English pipeline keeps the application usable even when
            # the model cannot be downloaded.
            return spacy.blank("en")


def safe_sent_tokenize(text):
    if not text.strip():
        return []
    try:
        return sent_tokenize(text)
    except Exception:
        return [s.strip() for s in text.splitlines() if s.strip()]


def safe_word_tokenize(text):
    if not text.strip():
        return []
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()


def safe_stopwords():
    try:
        return set(stopwords.words("english"))
    except Exception:
        return {
            "a", "an", "the", "and", "or", "but", "is", "are",
            "was", "were", "to", "of", "in", "on", "for", "at",
            "by", "with", "it", "this", "that", "as", "from"
        }


def analyze_text(text, nlp):
    start_time = time.perf_counter()

    sentences = safe_sent_tokenize(text)
    words = safe_word_tokenize(text)

    stop_words = safe_stopwords()

    # Only real alphanumeric tokens are counted as content/stopword tokens.
    alpha_numeric_words = [
        word for word in words if word.isalnum()
    ]

    filtered_words = [
        word for word in alpha_numeric_words
        if word.lower() not in stop_words
    ]

    removed_stopwords = [
        word for word in alpha_numeric_words
        if word.lower() in stop_words
    ]

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    try:
        doc = nlp(text)
    except Exception:
        doc = spacy.blank("en")(text)

    try:
        pos_tags = nltk.pos_tag(words)
    except Exception:
        pos_tags = []

    entities = [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]

    dependencies = [
        {
            "text": token.text,
            "dep": token.dep_,
            "head": token.head.text,
        }
        for token in doc
        if token.dep_ != "punct"
    ]

    try:
        noun_chunks = (
            [chunk.text for chunk in doc.noun_chunks]
            if doc.has_annotation("DEP")
            else []
        )
    except Exception:
        noun_chunks = []

    keyword_counts = Counter(
        word.lower()
        for word in filtered_words
        if len(word) > 2
    ).most_common(8)

    processing_time = time.perf_counter() - start_time

    return {
        "sentences": sentences,
        "words": words,
        "alpha_numeric_words": alpha_numeric_words,
        "filtered_words": filtered_words,
        "removed_stopwords": removed_stopwords,
        "stemmed_words": [
            stemmer.stem(word) for word in filtered_words
        ],
        "lemmatized_words": [
            lemmatizer.lemmatize(word) for word in filtered_words
        ],
        "pos_tags": pos_tags,
        "entities": entities,
        "dependencies": dependencies,
        "noun_chunks": noun_chunks,
        "keyword_counts": keyword_counts,
        "processing_time": processing_time,
    }


def build_entity_breakdown(entities):
    if not entities:
        return pd.DataFrame(columns=["entity_type", "count"])

    labels = [entity["label"] for entity in entities]
    breakdown = Counter(labels)

    return pd.DataFrame(
        {
            "entity_type": list(breakdown.keys()),
            "count": list(breakdown.values()),
        }
    )


def render_entity_donut_chart(entity_breakdown):
    if px is None:
        st.info("Plotly is not available. Install it with: pip install plotly")
        return

    if entity_breakdown.empty:
        st.info("No entities detected to display in the chart.")
        return

    fig = px.pie(
        entity_breakdown,
        names="entity_type",
        values="count",
        hole=0.52,
    )

    fig.update_traces(
        textinfo="percent+label",
        hovertemplate="%{label}: %{value}<extra></extra>",
    )

    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC"),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def build_download_payload(text, analysis):
    return {
        "text": text,
        "sentences": analysis["sentences"],
        "words": analysis["words"],
        "filtered_words": analysis["filtered_words"],
        "stemmed_words": analysis["stemmed_words"],
        "lemmatized_words": analysis["lemmatized_words"],
        "pos_tags": analysis["pos_tags"],
        "entities": analysis["entities"],
        "dependencies": analysis["dependencies"],
        "noun_chunks": analysis["noun_chunks"],
        "keyword_counts": analysis["keyword_counts"],
        "processing_time_seconds": analysis["processing_time"],
    }


def read_uploaded_file(uploaded):
    if uploaded is None:
        return None

    try:
        name = uploaded.name.lower()

        if name.endswith(".txt"):
            return uploaded.getvalue().decode("utf-8", errors="replace")

        if name.endswith(".pdf"):
            try:
                from PyPDF2 import PdfReader

                reader = PdfReader(uploaded)
                pages = [
                    page.extract_text() or ""
                    for page in reader.pages
                ]
                return "\n".join(pages)
            except ImportError:
                st.error("PDF support requires PyPDF2. Run: pip install PyPDF2")
                return None
            except Exception as exc:
                st.error(f"Could not read PDF: {exc}")
                return None

        if name.endswith(".docx"):
            try:
                import docx

                document = docx.Document(uploaded)
                return "\n".join(
                    paragraph.text
                    for paragraph in document.paragraphs
                )
            except ImportError:
                st.error(
                    "DOCX support requires python-docx. "
                    "Run: pip install python-docx"
                )
                return None
            except Exception as exc:
                st.error(f"Could not read DOCX: {exc}")
                return None

    except Exception as exc:
        st.error(f"Could not read uploaded file: {exc}")

    return None


def render_header():
    col1, col2 = st.columns([3.4, 1.6])

    with col1:
        st.markdown(
            """
            <div class="top-header">
                <div class="brand">
                    <div class="brand-icon">🧠</div>
                    <div>
                        <div class="brand-title">NLP Studio</div>
                        <div class="brand-subtitle">
                            Transform Raw Text into Actionable Insights
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.text_input(
            "Search",
            placeholder="Search metrics...",
            label_visibility="collapsed",
            key="metric_search",
        )


def render_hero_section():
    st.markdown(
        """
        <style>
        .hero-container {
            background: linear-gradient(135deg, rgba(0,201,150,.12) 0%, rgba(0,168,143,.08) 50%, rgba(0,230,176,.06) 100%);
            border: 1.5px solid rgba(0,201,150,.25);
            border-radius: 28px;
            padding: 64px 48px;
            margin: 24px 0 36px 0;
            text-align: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 16px 48px rgba(0,201,150,.08);
            backdrop-filter: blur(12px);
        }

        .hero-container::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(0,201,150,.12), transparent 70%);
            border-radius: 50%;
            pointer-events: none;
            filter: blur(60px);
        }

        .hero-container::after {
            content: '';
            position: absolute;
            bottom: -50%;
            left: -50%;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(0,168,143,.1), transparent 70%);
            border-radius: 50%;
            pointer-events: none;
            filter: blur(60px);
        }

        .hero-content {
            position: relative;
            z-index: 2;
        }

        .hero-icon {
            font-size: 72px;
            margin-bottom: 20px;
            display: inline-block;
            animation: float 3.5s ease-in-out infinite;
            filter: drop-shadow(0 8px 24px rgba(0,201,150,.3));
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px) scale(1); }
            50% { transform: translateY(-12px) scale(1.05); }
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 900;
            background: linear-gradient(135deg, #FFFFFF 0%, #B8E8E0 50%, #00c996 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 0 16px 0;
            line-height: 1.15;
            letter-spacing: -1px;
        }

        .hero-subtitle {
            font-size: 1.2rem;
            color: #B8C4D1;
            margin: 0 0 36px 0;
            max-width: 650px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.7;
        }

        .hero-cta {
            display: flex;
            gap: 14px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .hero-stat {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 14px 24px;
            background: rgba(0,61,77,.5);
            border: 1.5px solid rgba(0,201,150,.25);
            border-radius: 14px;
            backdrop-filter: blur(12px);
            margin: 8px;
            transition: all .3s ease;
            box-shadow: 0 4px 12px rgba(0,201,150,.08);
        }

        .hero-stat:hover {
            border-color: rgba(0,201,150,.4);
            background: rgba(0,61,77,.8);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,201,150,.15);
        }

        .hero-stat-value {
            font-weight: 900;
            background: linear-gradient(135deg, #00e6b0, #00c996);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 1.2rem;
        }

        .hero-stat-label {
            color: #B8C4D1;
            font-size: 0.9rem;
            font-weight: 600;
        }

        @media (max-width: 800px) {
            .hero-container {
                padding: 44px 24px;
            }
            .hero-title {
                font-size: 2.2rem;
            }
            .hero-subtitle {
                font-size: 1.05rem;
            }
            .hero-icon {
                font-size: 56px;
            }
        }
        </style>

        <div class="hero-container">
            <div class="hero-content">
                <div class="hero-icon">🧠</div>
                <h1 class="hero-title">Unlock the Power of NLP</h1>
                <p class="hero-subtitle">
                    Advanced text analysis, entity extraction, sentiment analysis, and intelligent insights 
                    powered by cutting-edge Natural Language Processing technology.
                </p>
                <div class="hero-cta">
                    <div class="hero-stat">
                        <span class="hero-stat-value">7+</span>
                        <span class="hero-stat-label">Analysis Tools</span>
                    </div>
                    <div class="hero-stat">
                        <span class="hero-stat-value">Real-time</span>
                        <span class="hero-stat-label">Processing</span>
                    </div>
                    <div class="hero-stat">
                        <span class="hero-stat-value">∞</span>
                        <span class="hero-stat-label">Scalability</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar(sample_text):
    with st.sidebar:
        st.markdown(
            """
            <div style="
                font-size:1.35rem;
                font-weight:850;
                color:#FFFFFF;
                margin:4px 0 14px 2px;
            ">
                Navigation
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_options = [
            "🧭 Overview",
            "📝 Text Analysis",
            "🔁 Pipeline",
            "🔗 Word2Vec",
            "📊 TF-IDF",
            "🏷️ Named Entities",
            "⚙️ Settings",
        ]

        # This is fully clickable and preserves the selected page.
        page = st.radio(
            "Navigation",
            nav_options,
            index=nav_options.index(
                st.session_state.get("page", "🧭 Overview")
            ),
            label_visibility="collapsed",
            key="page_selector",
        )

        st.session_state.page = page

        st.divider()

        if st.button(
            "📥  Load sample text",
            use_container_width=True,
            key="load_sample",
        ):
            st.session_state.text_input = sample_text
            st.rerun()

        st.markdown(
            "<div style='height:8px'></div>",
            unsafe_allow_html=True,
        )

        st.caption("Upload TXT, PDF, or DOCX to analyze")

        uploaded = st.file_uploader(
            "Upload file",
            type=["txt", "pdf", "docx"],
            label_visibility="collapsed",
            key="file_upload",
        )

        if uploaded is not None:
            # Avoid re-reading the same file on every rerun.
            current_file = f"{uploaded.name}:{uploaded.size}"

            if st.session_state.get("last_uploaded_file") != current_file:
                extracted_text = read_uploaded_file(uploaded)

                if extracted_text is not None:
                    st.session_state.text_input = extracted_text
                    st.session_state.last_uploaded_file = current_file
                    st.success("File loaded successfully.")


# ============================================================
# PAGE: OVERVIEW
# ============================================================
def render_overview(text, analysis):
    st.markdown('<div class="section-title">At a glance</div>', unsafe_allow_html=True)

    preview = text[:700] + ("..." if len(text) > 700 else "")

    st.markdown(
        f"""
        <div class="glass fade">
            <div style="
                font-weight:750;
                color:#FFFFFF;
                margin-bottom:8px;
            ">
                Selected Text Preview
            </div>
            <div class="subtle">
                {preview.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Entity Breakdown</div>',
        unsafe_allow_html=True,
    )

    entity_breakdown = build_entity_breakdown(analysis["entities"])
    render_entity_donut_chart(entity_breakdown)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Download Results</div>',
        unsafe_allow_html=True,
    )

    payload = build_download_payload(text, analysis)
    json_payload = json.dumps(payload, indent=2, ensure_ascii=False)

    if analysis["entities"]:
        csv_payload = pd.DataFrame(analysis["entities"]).to_csv(index=False)
    else:
        csv_payload = "text,label\n"

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "⬇️ Download JSON Analysis",
            data=json_payload,
            file_name="nlp_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            "⬇️ Download Entities CSV",
            data=csv_payload,
            file_name="entities.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# PAGE: TEXT ANALYSIS
# ============================================================
def render_text_analysis(analysis):
    st.markdown(
        '<div class="section-title">Sentence Segmentation</div>',
        unsafe_allow_html=True,
    )

    for i, sentence in enumerate(analysis["sentences"], 1):
        st.markdown(
            f"""
            <div class="glass" style="margin-bottom:8px;">
                <b>Sentence {i}</b>
                <div class="subtle" style="margin-top:6px;">
                    {sentence}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Tokenization & Cleanup</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("Original Tokens")
        st.code(analysis["words"])

    with col2:
        st.write("Filtered Tokens")
        st.code(analysis["filtered_words"])

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Lemmatized & Stemmed Output</div>',
        unsafe_allow_html=True,
    )

    col3, col4 = st.columns(2)

    with col3:
        st.write("Stemmed Words")
        st.code(analysis["stemmed_words"])

    with col4:
        st.write("Lemmatized Words")
        st.code(analysis["lemmatized_words"])


# ============================================================
# PAGE: PIPELINE
# ============================================================
def render_pipeline(analysis):
    st.markdown(
        '<div class="section-title">NLP Pipeline</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Part-of-Speech Tagging")
        if analysis["pos_tags"]:
            pos_df = pd.DataFrame(
                analysis["pos_tags"],
                columns=["word", "tag"],
            )
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.info("POS tags are not available.")

    with col2:
        st.markdown("#### Top Keywords")
        if analysis["keyword_counts"]:
            df_kw = pd.DataFrame(
                analysis["keyword_counts"],
                columns=["term", "count"],
            )
            st.bar_chart(
                df_kw.set_index("term"),
                use_container_width=True,
            )
        else:
            st.info("No sufficient keywords found.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Dependency Parse")

    if analysis["dependencies"]:
        st.dataframe(
            pd.DataFrame(analysis["dependencies"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Dependency parsing is unavailable. "
            "Install the spaCy English model for full parsing."
        )

    st.markdown("#### Noun Phrase Chunks")

    if analysis["noun_chunks"]:
        st.code(analysis["noun_chunks"])
    else:
        st.info("No noun chunks available.")


# ============================================================
# PAGE: WORD2VEC
# ============================================================
def render_word2vec(analysis):
    st.markdown(
        '<div class="section-title">🔗 Word2Vec</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass">
            Word2Vec normally requires a trained embedding model.
            This page shows the cleaned vocabulary that can be used
            as input for Word2Vec training.
        </div>
        """,
        unsafe_allow_html=True,
    )

    words = sorted(set(
        word.lower()
        for word in analysis["filtered_words"]
        if word.isalpha()
    ))

    if words:
        st.markdown("#### Vocabulary")
        vocab_df = pd.DataFrame(
            {"word": words, "frequency": [
                analysis["filtered_words"].count(word)
                for word in words
            ]}
        )
        st.dataframe(
            vocab_df.sort_values(
                "frequency",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Enter more text to build a vocabulary.")


# ============================================================
# PAGE: TF-IDF
# ============================================================
def render_tfidf(analysis):
    st.markdown(
        '<div class="section-title">📊 TF-IDF</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass">
            TF-IDF gives higher importance to words that are frequent
            in a document but less common across documents.
            The table below calculates TF-IDF across the sentences
            in the current text.
        </div>
        """,
        unsafe_allow_html=True,
    )

    documents = [
        sentence.lower()
        for sentence in analysis["sentences"]
        if sentence.strip()
    ]

    if len(documents) < 2:
        st.info("Enter at least two sentences to calculate TF-IDF.")
        return

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
        )

        matrix = vectorizer.fit_transform(documents)
        terms = vectorizer.get_feature_names_out()

        tfidf_df = pd.DataFrame(
            matrix.toarray(),
            columns=terms,
            index=[
                f"Sentence {i + 1}"
                for i in range(len(documents))
            ],
        )

        st.dataframe(
            tfidf_df.round(3),
            use_container_width=True,
        )

        avg_scores = tfidf_df.mean().sort_values(
            ascending=False
        ).head(10)

        st.markdown("#### Top TF-IDF Terms")

        if not avg_scores.empty:
            st.bar_chart(avg_scores, use_container_width=True)

    except ImportError:
        st.error(
            "scikit-learn is required for TF-IDF. "
            "Run: pip install scikit-learn"
        )
    except ValueError:
        st.info("Not enough meaningful words to calculate TF-IDF.")


# ============================================================
# PAGE: NAMED ENTITIES
# ============================================================
def render_entities(analysis):
    st.markdown(
        '<div class="section-title">🏷️ Named Entities Detected</div>',
        unsafe_allow_html=True,
    )

    if analysis["entities"]:
        entity_df = pd.DataFrame(analysis["entities"])
        st.dataframe(
            entity_df,
            use_container_width=True,
            hide_index=True,
        )

        breakdown = build_entity_breakdown(analysis["entities"])
        st.markdown("#### Entity Summary")
        st.dataframe(
            breakdown,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No named entities detected. "
            "Make sure the spaCy English model is installed."
        )


# ============================================================
# PAGE: SETTINGS
# ============================================================
def render_settings():
    st.markdown(
        '<div class="section-title">⚙️ Settings</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass">
            <b>NLP Studio Configuration</b>
            <div class="subtle" style="margin-top:8px;">
                Use the controls below to manage the application.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Display")

    show_processing = st.checkbox(
        "Show processing time",
        value=True,
    )

    if show_processing:
        st.info("Processing time is shown in the Overview Stats card.")

    st.markdown("#### Current Environment")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Python", sys.version.split()[0])

    with col2:
        st.metric("Streamlit", st.__version__)

    with col3:
        st.metric(
            "spaCy",
            spacy.__version__,
        )

    st.markdown("#### Required Packages")
    st.code(
        """pip install streamlit nltk pandas spacy plotly scikit-learn PyPDF2 python-docx
python -m spacy download en_core_web_sm""",
        language="bash",
    )


# ============================================================
# MAIN APP
# ============================================================
def main():
    apply_custom_css()
    ensure_nltk_data()

    nlp = load_spacy_model()

    sample_text = (
        "Natural Language Processing is a field of Artificial Intelligence.\n"
        "It helps computers understand human language.\n"
        "Dr. Suhashini Chaurasia lives in India and works at RBU."
    )

    if "text_input" not in st.session_state:
        st.session_state.text_input = sample_text

    if "page" not in st.session_state:
        st.session_state.page = "🧭 Overview"

    render_sidebar(sample_text)
    render_header()
    render_hero_section()

    # ---------------- INPUT ----------------
    st.markdown(
        '<div class="section-title">Input Text</div>',
        unsafe_allow_html=True,
    )

    text = st.text_area(
        "Input text",
        value=st.session_state.text_input,
        height=190,
        placeholder="Paste or type some text to analyze...",
        label_visibility="collapsed",
    )

    # Keep state synchronized with the text area.
    st.session_state.text_input = text

    st.markdown(
        f"""
        <div style="
            text-align:right;
            color:#AFC0D8;
            margin-top:6px;
            font-size:.82rem;
        ">
            {len(text):,} characters
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- ANALYSIS ----------------
    with st.spinner("Analyzing your text..."):
        analysis = analyze_text(text, nlp)

    # ---------------- STATS ----------------
    st.markdown(
        '<div class="section-title">Overview Stats</div>',
        unsafe_allow_html=True,
    )

    stats = [
        (
            "Sentences",
            len(analysis["sentences"]),
            "Sentences",
            "🧾",
        ),
        (
            "Words",
            len(analysis["alpha_numeric_words"]),
            "Word tokens",
            "🔤",
        ),
        (
            "Filtered",
            len(analysis["filtered_words"]),
            "After stopwords",
            "🧹",
        ),
        (
            "Unique",
            len(set(
                word.lower()
                for word in analysis["filtered_words"]
            )),
            "Unique terms",
            "🔎",
        ),
        (
            "Entities",
            len(analysis["entities"]),
            "Named entities",
            "🏷️",
        ),
        (
            "Vocab",
            len(set(
                word.lower()
                for word in analysis["alpha_numeric_words"]
                if word.isalpha()
            )),
            "Vocabulary size",
            "📚",
        ),
        (
            "Time",
            f"{analysis['processing_time']:.2f}s",
            "Processing time",
            "⏱️",
        ),
        (
            "Stopwords",
            len(analysis["removed_stopwords"]),
            "Removed",
            "🚮",
        ),
    ]

    cols = st.columns(4)

    for i, stat in enumerate(stats):
        with cols[i % 4]:
            render_metric_card(
                stat[0],
                stat[1],
                stat[2],
                icon=stat[3],
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- ROUTING ----------------
    page = st.session_state.page

    if page == "🧭 Overview":
        render_overview(text, analysis)

    elif page == "📝 Text Analysis":
        render_text_analysis(analysis)

    elif page == "🔁 Pipeline":
        render_pipeline(analysis)

    elif page == "🔗 Word2Vec":
        render_word2vec(analysis)

    elif page == "📊 TF-IDF":
        render_tfidf(analysis)

    elif page == "🏷️ Named Entities":
        render_entities(analysis)

    elif page == "⚙️ Settings":
        render_settings()


if __name__ == "__main__":
    main()