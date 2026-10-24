"""
modelA.py  -  OCR stage (PaddleOCR PP-OCRv5)

Image  ->  list of text items with positions  ->  text block for the LLM (Model B)

Usage in app.py:
    from modelA import run_ocr
    ocr_text = run_ocr("bill.jpg")
"""

from paddleocr import PaddleOCR
from src.config import load_config
import os
import logging

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger=logging.getLogger("model A")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"modelA.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


_ocr = None  # loaded once, reused for every image


def load_ocr():
    try:
        global _ocr
        if _ocr is None:
            cfg = load_config()["ocr"]
            _ocr = PaddleOCR(
                text_detection_model_name=cfg["detection_model"],
                text_recognition_model_name=cfg["recognition_model"],
                use_doc_orientation_classify=cfg["use_doc_orientation_classify"],
                use_doc_unwarping=cfg["use_doc_unwarping"],
                use_textline_orientation=cfg["use_textline_orientation"],
            )
        logger.debug("ocr model loaded")
        return _ocr
        
    except Exception as e:
        logger.error("ocr model failed to load")




def extract_ocr_items(image_path):
    """Run OCR and return items sorted top-to-bottom, then left-to-right.

    Each item: {"text": str, "x": center_x, "y": center_y, "box": [x1, y1, x2, y2]}
    """
    try:
        logger.debug("ocr items fetching started")
        ocr = load_ocr()
        result = ocr.predict(image_path)

        items = []
        for res in result:
            texts = res["rec_texts"]
            boxes = res["rec_boxes"]
            for text, box in zip(texts, boxes):
                x1, y1, x2, y2 = map(int, box)
                items.append({
                    "text": text,
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "box": [x1, y1, x2, y2],
                })

        items.sort(key=lambda i: (i["y"], i["x"]))
        logger.debug("ocr items succesfully fetched from image")
        return items
    except Exception as e:
        logger.error("failed to extract ocr items from image:%s",e)


def items_to_text(ocr_items):
    """Convert OCR items into the 'TEXT | X | Y' format Model B expects."""  
    try:
        logger.debug("items to text conversion started")
        ocr_text_with_position = ""

        for item in ocr_items:
            ocr_text_with_position += (
                f"TEXT: {item['text']} | "
                f"X: {item['x']:.0f} | "
                f"Y: {item['y']:.0f}\n"
            )
        logger.debug("items to text conversion completed")
        return ocr_text_with_position
    except Exception as e:
        logger.error("converting ocr items to text failed due to %s",e)

def run_ocr(image_path):
    """Main function for app.py: image path -> positioned OCR text string."""
    return items_to_text(extract_ocr_items(image_path))