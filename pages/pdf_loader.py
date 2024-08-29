import shutil
from pathlib import Path

import streamlit as st

from utility import handler, loader

TEMP_DIR = "temp_dir"

sess = st.session_state
if TEMP_DIR not in sess:
    temp_dir = Path("temp")
    if temp_dir.exists():
        shutil.rmtree(str(temp_dir))
    temp_dir.mkdir(exist_ok=True, parents=True)
    sess[TEMP_DIR] = temp_dir


st.title("PDF Loader")

loader.pdf_controller_section(
    session=sess,
    temp_dir=sess[TEMP_DIR],
    uploader_name="pdf_loader",
    converter_name="pdf_conveter",
    converter_options=handler.SUP_CONVS,
    start_function=loader.load_pdf_and_split_with_haystack,
)
