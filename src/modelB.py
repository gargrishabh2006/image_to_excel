"""
modelB.py  -  LLM extraction stage (Llama-3.2-3B-Instruct)

OCR text (with positions)  +  user fields  ->  dict {field: value}

Usage in app.py:
    from modelB import extract_fields
    data = extract_fields(ocr_text, ["Qty", "Total", "Bill To"])

Set your Hugging Face token as an environment variable (do NOT hardcode it):
    export HF_TOKEN=hf_xxxxxxxx
Llama 3.2 is a gated model: accept its license on huggingface.co first.
"""
from dotenv import load_dotenv
import ast
import json
import os
import re
import torch
from huggingface_hub import login
from transformers import pipeline
from src.config import load_config
load_dotenv()

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

_pipe = None  # loaded once, reused for every bill

SYSTEM_PROMPT = """
You are an invoice information extraction assistant.

You will receive OCR text along with the X and Y position of each
detected text element.

The X coordinate represents the horizontal position on the invoice.
The Y coordinate represents the vertical position.

IMPORTANT RULES:

1. Use the spatial position of OCR elements to determine which
   column and row they belong to.

2. Do NOT assume that numbers appearing first in OCR text are Qty.

3. For table data, identify the column headers such as:
   Particulars, Qty, Rate, Amount, Tax, etc.

4. A number below the Qty column must be treated as Qty.
   A number below the Rate column must be treated as Rate.
   A number below the Amount column must be treated as Amount.

5. Preserve numeric values exactly as detected whenever possible.

6. Do not swap Qty, Rate, or Amount based only on their numerical values.

7. Correct obvious OCR spelling errors only when the correction is
   clear from context.

8. Do not modify GSTINs.
9. Do not modify invoice numbers.
10. Do not modify dates.
11. Do not modify amounts unless the OCR clearly contains an error.

12. Return output as dictionary with correct opening and closing brackets and key value format.
13. Do not provide explanations or any other text.
14. each field will have only 1 value (numeric)
"""


def load_llm():
    global _pipe
    if _pipe is None:
        cfg = load_config()["llm"]
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
        _pipe = pipeline(
            "text-generation",
            model=cfg["model_id"],
            dtype=getattr(torch, cfg["dtype"]),
            device_map=cfg["device_map"],
        )
    return _pipe
 
 


def _parse_dict(raw):
    """Safely turn the LLM's reply into a dict (handles extra text / code fences)."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No dictionary found in model output:\n{raw}")
    text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def extract_fields(ocr_text, fields, max_new_tokens=256):
    """Main function for app.py.

    ocr_text : string returned by modelA.run_ocr()
    fields   : list of column names the user wants, e.g. ["Qty", "Total", "Bill To"]
    returns  : dict with exactly those keys (missing ones are None)
    """
    pipe = load_llm()

    user_prompt = f"""
        Fields to extract:
        {', '.join(fields)}

        OCR DATA WITH POSITIONS:
        {ocr_text}
        """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    outputs = pipe(messages, max_new_tokens=max_new_tokens)
    raw = outputs[0]["generated_text"][-1]["content"]

    parsed = _parse_dict(raw)
    # keep only requested fields, in the user's order
    return {field: parsed.get(field) for field in fields}