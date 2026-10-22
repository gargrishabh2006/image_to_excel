import os
import tempfile

import pandas as pd
import streamlit as st
from src.modelB import extract_fields, load_llm
from src.excel import save_to_excel
from src.config import load_config
from src.modelA import load_ocr, run_ocr

config = load_config()
EXCEL_FILE = config["excel"]["output_file"]

# Load all models once (cached for the whole Streamlit server session)
@st.cache_resource
def load_models():
    load_ocr()
    load_llm()
    return True
 
 
with st.spinner("Loading models, please wait..."):
    load_models()

 
st.title("Image to Excel")
st.write("Upload bill images and get all the data in one Excel file.")

# 1. Fields (set once)
fields_input = st.text_input("Fields to extract (comma separated)", "Qty, Total, Bill To")
fields = [f.strip() for f in fields_input.split(",") if f.strip()]

# 2. Upload bills
files = st.file_uploader(
    "Upload bill images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# 3. Process
if st.button("Extract") and files and fields:
    rows = []
    progress = st.progress(0)

    for i, f in enumerate(files):
        # save upload to a temp file because OCR needs a file path
        suffix = os.path.splitext(f.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getbuffer())
            tmp_path = tmp.name

        try:
            ocr_text = run_ocr(tmp_path)                 # Model A
            data = extract_fields(ocr_text, fields)      # Model B
            save_to_excel(data, fields, EXCEL_FILE)      # Excel
            rows.append({"File": f.name, **data})
        except Exception as e:
            st.error(f"{f.name}: failed ({e})")
        finally:
            os.remove(tmp_path)

        progress.progress((i + 1) / len(files))

    if rows:
        st.success("Done!")
        st.dataframe(pd.DataFrame(rows))

# 4. Download Excel
if os.path.exists(EXCEL_FILE):
    with open(EXCEL_FILE, "rb") as f:
        st.download_button("Download Excel", f, file_name="bills.xlsx")

    if st.button("Clear Excel file (start new day)"):
        os.remove(EXCEL_FILE)
        st.rerun()