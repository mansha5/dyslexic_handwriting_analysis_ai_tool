import textstat
import pytesseract
from PIL import Image


def calculate_readability_scores(text):
    """
    Calculate multiple readability metrics using the `textstat` library.
    These scores help measure how easy or difficult the text is to read.
    """
    try:
        return {
            "Flesch Reading Ease": round(textstat.flesch_reading_ease(text), 2),
            "Flesch-Kincaid Grade": round(textstat.flesch_kincaid_grade(text), 2),
            "SMOG Index": round(textstat.smog_index(text), 2),
            "Automated Readability Index": round(textstat.automated_readability_index(text), 2),
            "Dale-Chall Score": round(textstat.dale_chall_readability_score(text), 2)
        }
    except Exception:
        return {}


def extract_text_from_image(image_path):
    """
    Use Tesseract OCR to extract readable text from an uploaded image.
    """
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {e}"