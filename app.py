# ============================================
# 🧠 LearnBridge - Streamlit App (Mac + Windows)
# ============================================

import streamlit as st

st.set_page_config(page_title="LearnBridge", layout="wide")

import pandas as pd
import plotly.graph_objects as go
import re
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import torch
import zipfile
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from difflib import SequenceMatcher
from datetime import datetime
import platform
import os

# =====================
# PATHS & SETUP
# =====================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "text_correction_hybrid"
FEEDBACK_PATH = BASE_DIR / "output" / "processed_text.txt"
FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

#  Detect OS and set Tesseract path
import shutil

tesseract_path = shutil.which("tesseract")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    st.error("Tesseract not found. Run: brew install tesseract")

# =====================
#  IMAGE PREPROCESSOR
# =====================
def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    img = pil_img.copy()
    w, h = img.size
    if h > w and (h / w) > 1.2:
        img = img.rotate(-90, expand=True)
    img = img.convert("L")
    base_w = 1200
    if img.width < base_w:
        ratio = base_w / img.width
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=2)
    mean = sum(img.getdata()) / (img.width * img.height)
    if mean < 120:
        img = ImageOps.invert(img)
    img = img.filter(ImageFilter.SHARPEN)
    return img

# =====================
#  MODEL LOADING
# =====================
@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    return tokenizer, model

tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# =====================
# 🔤 OCR & UTILS
# =====================
def ocr_image_to_text(pil_img: Image.Image) -> str:
    try:
        img = preprocess_for_ocr(pil_img)
        text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
        return text.strip()
    except Exception:
        return ""

def compute_text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# =====================
#  SCORING FUNCTIONS
# =====================
def text_features(text: str):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    if len(words) == 0:
        return {"n_words": 0, "n_sentences": 0, "avg_sentence_len": 0.0,
                "avg_word_len": 0.0, "type_token_ratio": 0.0, "complex_word_frac": 0.0}
    n_words = len(words)
    n_sentences = max(1, len(sentences))
    avg_sentence_len = n_words / n_sentences
    avg_word_len = sum(len(w) for w in words) / n_words
    type_token_ratio = len(set(words)) / n_words
    complex_word_frac = sum(1 for w in words if len(w) >= 7) / n_words
    return {"n_words": n_words, "n_sentences": n_sentences, "avg_sentence_len": avg_sentence_len,
            "avg_word_len": avg_word_len, "type_token_ratio": type_token_ratio, "complex_word_frac": complex_word_frac}

def compute_scores_from_features(features: dict, error_percent: float = 0.0):
    grammar = max(0.0, 100.0 - error_percent)
    ttr = features.get("type_token_ratio", 0.0)
    avg_w = features.get("avg_word_len", 0.0)
    avg_w_norm = min(1.0, max(0.0, (avg_w - 2.0) / 6.0))
    vocab = (0.6 * ttr + 0.4 * avg_w_norm) * 100.0
    vocab = max(0.0, min(100.0, vocab))
    asl = features.get("avg_sentence_len", 0.0)
    asl_norm = min(1.0, max(0.0, (asl - 3.0) / 17.0))
    complex_frac = features.get("complex_word_frac", 0.0)
    expression = (0.6 * asl_norm + 0.4 * complex_frac) * 100.0
    expression = max(0.0, min(100.0, expression))
    return round(grammar, 2), round(vocab, 2), round(expression, 2)

def compute_overall_score(grammar, vocab, expression):
    return round(0.4 * grammar + 0.3 * vocab + 0.3 * expression, 2)

def aggregate_score_to_grade(overall_score: float):
    if overall_score >= 85:
        return "Grade 7+", 7
    if overall_score >= 70:
        return "Grade 5–6", 5
    if overall_score >= 55:
        return "Grade 3–4", 3
    return "Grade 1–2", 1

def simplify_with_model(text, max_input_len=512, max_out_len=256):
    if not text or not text.strip():
        return ""
    prefix = "simplify: " + text.strip()
    inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=max_input_len).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=max_out_len, num_beams=4, early_stopping=True)
    simplified = tokenizer.decode(out[0], skip_special_tokens=True)
    return simplified

# =====================
#  Parent Report
# =====================
def generate_parent_report(grade_o, grade_c, grammar_o, vocab_o, grammar_c, vocab_c, error_percent):
    lines = []
    lines.append(f"1️⃣ The AI analysis shows your child’s writing currently resembles **{grade_o} level**, improving to **{grade_c} level** after corrections.")
    lines.append(f"2️⃣ Grammar improved from {grammar_o}% to {grammar_c}%, and vocabulary from {vocab_o}% to {vocab_c}%.")
    lines.append(f"3️⃣ About **{error_percent}%** of the text required correction — this indicates minor gaps in spelling or structure.")
    lines.append("4️⃣ Focus on reinforcing correct sentence patterns and common grammar rules in daily writing tasks.")
    lines.append("5️⃣ Vocabulary can be improved by encouraging your child to read short illustrated storybooks.")
    lines.append("6️⃣ Ask your child to retell a story in 4–5 sentences, emphasizing punctuation and word variety.")
    lines.append("7️⃣ Small improvements each week (5–10% grammar gain) indicate steady progress.")
    lines.append("8️⃣ Create a word wall at home — write new words and use them in simple sentences.")
    lines.append("9️⃣ Track 3 writing samples over the next month to confirm consistent improvement.")
    lines.append("🔟 Celebrate small milestones — appreciation builds motivation for better writing.")
    return "\n\n".join(lines)

# =====================
#  STREAMLIT UI
# =====================

st.title("🧠 LearnBridge - your child's pathway to clarity")
st.caption("Upload a handwriting image or paste text. The app estimates grade level and provides an easy-to-understand parent report.")

uploaded = st.file_uploader("Upload handwriting image or .txt", type=["png", "jpg", "jpeg", "txt"])
pasted = st.text_area("Or paste text here:", height=150)
process = st.button("Analyze Writing")

# Run analysis only when clicked
if process:
    if pasted.strip():
        input_text = pasted.strip()
        st.success("Using pasted text as input.")
        uploaded_image = None
    elif uploaded:
        if uploaded.type.startswith("image"):
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)
            text = ocr_image_to_text(img)
            if not text:
                st.error("❌ No readable text detected. Try a clearer image.")
                st.stop()
            input_text = text
            st.success("✅ Text extracted successfully from image.")
        else:
            input_text = uploaded.read().decode("utf-8")
            st.success("Using uploaded text file.")
    else:
        st.warning("Please upload or paste text.")
        st.stop()

    # --- Run model + compute scores ---
    corrected_text = simplify_with_model(input_text)
    with st.expander("View Original Text"):
        st.text_area("Original Extracted Text", input_text, height=200)

    with st.expander("View Corrected Text (AI Simplified)", expanded=False):
        st.text_area("AI Corrected Text", corrected_text, height=200)
    feat_o = text_features(input_text)
    feat_c = text_features(corrected_text)
    sim = compute_text_similarity(input_text, corrected_text)
    err = round((1 - sim) * 100, 2)

    g_o, v_o, e_o = compute_scores_from_features(feat_o, err)
    g_c, v_c, e_c = compute_scores_from_features(feat_c, 0)
    overall_o = compute_overall_score(g_o, v_o, e_o)
    overall_c = compute_overall_score(g_c, v_c, e_c)
    grade_o, _ = aggregate_score_to_grade(overall_o)
    grade_c, _ = aggregate_score_to_grade(overall_c)

    st.markdown("### 📊 Grammar & Vocabulary Scores Comparison")
    col1, col2 = st.columns(2)

    # ========== GRAMMAR ==========
    with col1:
        st.subheader("📝 Grammar")
        g1, g2 = st.columns(2)

        fig_g_before = go.Figure(go.Indicator(
            mode="gauge+number",
            value=g_o,
            title={'text': "Before Correction", 'font': {'size': 16}},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF6B6B"}}
        ))
        g1.plotly_chart(fig_g_before, use_container_width=True)

        fig_g_after = go.Figure(go.Indicator(
            mode="gauge+number",
            value=g_c,
            title={'text': "After Correction", 'font': {'size': 16}},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FFB6B6"}}
        ))
        g2.plotly_chart(fig_g_after, use_container_width=True)

    # ========== VOCABULARY ==========
    with col2:
        st.subheader("📚 Vocabulary")
        v1, v2 = st.columns(2)

        fig_v_before = go.Figure(go.Indicator(
            mode="gauge+number",
            value=v_o,
            title={'text': "Before Correction", 'font': {'size': 16}},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FFD93D"}}
        ))
        v1.plotly_chart(fig_v_before, use_container_width=True)

        fig_v_after = go.Figure(go.Indicator(
            mode="gauge+number",
            value=v_c,
            title={'text': "After Correction", 'font': {'size': 16}},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FFF3B0"}}
        ))
        v2.plotly_chart(fig_v_after, use_container_width=True)

    st.markdown("## 👪 Parent Feedback Summary")
    feedback = generate_parent_report(grade_o, grade_c, g_o, v_o, g_c, v_c, err)
    st.info(feedback)