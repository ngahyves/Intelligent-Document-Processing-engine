#src/ocr/post_process
import re

def clean_text(text: str) -> str:
    """
    cleaning raw text from OCR.
    """
    if not text:
        return ""

    # 1. Delete special characters
    text = re.sub(r'[^\w\s.,€$]', '', text)
    
    # 2. Replace lines break by spaces
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    # 3. Delete multiples spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text