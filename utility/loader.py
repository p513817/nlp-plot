from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utility import handler


def find_pdf(path: str):
    return list(Path(path).glob("*.pdf"))


def pdf_uploader_event(temp_dir: Path, uploader_name: str, session):
    uploaded_files = session[uploader_name]

    uploaded_pdfs = set()
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        uploaded_pdfs.add(file_name)

        file_path = temp_dir / file_name
        if file_path.exists():
            continue

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.toast(f"Save PDF : {file_name}")

    for pdf_path in find_pdf(temp_dir):
        pdf_name = pdf_path.name
        if pdf_name in uploaded_pdfs:
            continue
        pdf_path.unlink()
        st.toast(f"Remove Unused PDF: {pdf_name}")


def get_conv_clean_splitter(
    session,
    converter_name: str,
    split_by_name: str,
    split_leng_name: str,
    split_overlap_name: str,
    split_thres_name: str,
):
    converters = handler.get_converters()
    converter_func = converters.get(session[converter_name], None)
    if converter_func is None:
        raise KeyError(
            f"Expected converter name is {','.join(converters.keys())}, but get {session[converter_name]}"
        )
    converter = converter_func()
    cleaner = handler.get_cleaner()
    splitter = handler.get_splitter(
        session[split_by_name],
        session[split_leng_name],
        session[split_overlap_name],
        session[split_thres_name],
    )
    return (converter, cleaner, splitter)


def process_with_haystack(pdf_path_list, converter, cleaner, splitter):
    documents_org = converter.run(
        sources=pdf_path_list, meta={"createdAt": datetime.now().isoformat()}
    )
    documents_clean = cleaner.run(documents_org["documents"])
    documents_split = splitter.run(documents_clean["documents"])
    return documents_split


def load_pdf_and_split_with_haystack(
    converter, cleaner, splitter, pdf_path_list: list, **kwargs
):
    for pdf_path in pdf_path_list:
        documents = process_with_haystack(
            pdf_path_list=[pdf_path],
            converter=converter,
            cleaner=cleaner,
            splitter=splitter,
        )["documents"]

        data_list = []
        for document in documents:
            split_id = document.meta["split_id"]
            split_idx_start = document.meta["split_idx_start"]
            page_number = document.meta["page_number"]
            content = document.content
            # Append to data list
            data_list.append(
                {
                    "split_id": split_id,
                    "split_idx_start": split_idx_start,
                    "page_number": page_number,
                    "content": content,
                }
            )
        with st.expander(label=pdf_path.name):
            # Convert to DataFrame
            df = pd.DataFrame(data_list)
            df = df.set_index(df.columns[0])
            st.dataframe(df)


def load_split_embed_with_haystack(
    converter, cleaner, splitter, pdf_path_list: list, **kwargs
):
    pdf_data = defaultdict(list)
    for pdf_path in pdf_path_list:
        documents = process_with_haystack(
            pdf_path_list=[pdf_path],
            converter=converter,
            cleaner=cleaner,
            splitter=splitter,
        )["documents"]

        sentences, colors = [], []
        for document in documents:
            split_id = document.meta["split_id"]
            split_idx_start = document.meta["split_idx_start"]
            page_number = document.meta["page_number"]
            content = document.content
            content = f"{pdf_path.name}: {content}"
            # Append to data list
            pdf_data[pdf_path].append(
                {
                    "split_id": split_id,
                    "split_idx_start": split_idx_start,
                    "page_number": page_number,
                    "content": content,
                }
            )
            sentences.append(content)
            colors.append(handler.generate_random_color())

        with st.expander(label=pdf_path.name):
            # Convert to DataFrame
            df = pd.DataFrame(pdf_data[pdf_path])
            df = df.set_index(df.columns[0])

            # Apply the color styles to the DataFrame
            def color_rows(row):
                return ["background-color: {}".format(colors[row.name])] * len(row)

            styled_df = df.style.apply(color_rows, axis=1)
            st.dataframe(styled_df)

            # PCA
            sentence_embeddings = handler.sentence_to_embedding(sentences=sentences)
            sentence_embeddings_2d = handler.sentence_embedding_to_2d(
                sentence_embeddings=sentence_embeddings
            )

            # Figure
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sentence_embeddings_2d[:, 0],
                    y=sentence_embeddings_2d[:, 1],
                    mode="markers",
                    text=sentences,
                    name="Sentence",
                    marker=dict(color=colors, colorscale="Viridis", line_width=1),
                )
            )

            fig.update_traces(marker=dict(size=15), textposition="top center")
            fig.update_layout(
                title="2D PCA of Sentence Embeddings",
            )

            # Display the plot
            st.plotly_chart(fig)


def compare_pdf_embedding(converter, cleaner, splitter, pdf_path_list: list, **kwargs):
    pdf_data = defaultdict(list)
    pdf_sentences = defaultdict(list)
    pdf_split_id = defaultdict(list)
    pdf_sentences_length = defaultdict(int)
    pdf_colors = defaultdict(any)

    total_pdf_sentence = []

    for pdf_path in pdf_path_list:
        documents = process_with_haystack(
            pdf_path_list=[pdf_path],
            converter=converter,
            cleaner=cleaner,
            splitter=splitter,
        )["documents"]

        cur_color = handler.generate_random_color()

        for document in documents:
            split_id = document.meta["split_id"]
            split_idx_start = document.meta["split_idx_start"]
            page_number = document.meta["page_number"]
            content = document.content
            content = f"{pdf_path.stem}: {content}"

            # Append to data list
            pdf_data[pdf_path].append(
                {
                    "split_id": split_id,
                    "split_idx_start": split_idx_start,
                    "page_number": page_number,
                    "content": content,
                }
            )
            pdf_split_id[pdf_path].append(split_id)
            pdf_sentences[pdf_path].append(content)
            pdf_sentences_length[pdf_path] += 1

        with st.expander(label=pdf_path.name):
            # Convert to DataFrame
            df = pd.DataFrame(pdf_data[pdf_path])
            df = df.set_index(df.columns[0])

            # Apply the color styles to the DataFrame
            def color_rows(row):
                return ["background-color: {}".format(cur_color)] * len(row)

            styled_df = df.style.apply(color_rows, axis=1)
            st.dataframe(styled_df)

        total_pdf_sentence += pdf_sentences[pdf_path]
        pdf_colors[pdf_path] = cur_color

    # PCA
    with st.spinner("Load Figure ..."):
        pdf_embeddings = handler.sentence_to_embedding(sentences=total_pdf_sentence)
        sentence_embeddings_2d = handler.sentence_embedding_to_2d(
            sentence_embeddings=pdf_embeddings
        )

        # Figure
        fig = go.Figure()

        # Split PCA Data
        start_idx = 0
        for pdf_path, pdf_length in pdf_sentences_length.items():
            pdf_embed = sentence_embeddings_2d[start_idx : start_idx + pdf_length]
            start_idx = pdf_length

            fig.add_trace(
                go.Scatter(
                    x=pdf_embed[:, 0],
                    y=pdf_embed[:, 1],
                    mode="markers",
                    text=pdf_split_id[pdf_path],
                    name=str(pdf_path.name),
                    marker=dict(color=pdf_colors[pdf_path]),
                )
            )

        fig.update_traces(marker=dict(size=15), textposition="top center")
        fig.update_layout(
            title="2D PCA of Sentence Embeddings",
        )

        # Display the plot
        st.plotly_chart(fig)


def pdf_controller_section(
    session,
    temp_dir: Path,
    uploader_name: str,
    converter_name: str,
    converter_options: list,
    split_by_name: str = "split_by_selectbox",
    split_leng_name: str = "split_length_slider",
    split_overlap_name: str = "split_overlap_slider",
    split_thres_name: str = "split_thres_name",
    start_function: callable = load_pdf_and_split_with_haystack,
    **kwargs,
):
    """
    This section includes:
        1. File_uploader with PDF format and support multiple files
        2. Advance setting ( Splitter )
            a. Selectbox: Split by word, sentence, passafe, page
            b. Slider: Split Length
            c. Slider: Split Overlap
            d. Slider: Split Threshold

    """
    # File uploader
    st.file_uploader(
        "Upload a PDF file",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
        on_change=pdf_uploader_event,
        key=uploader_name,
        kwargs={
            "temp_dir": Path(temp_dir),
            "uploader_name": uploader_name,
            "session": session,
        },
    )

    with st.expander(label="Advance Setting", expanded=False):
        # Select PDF loader
        st.selectbox(
            label="Select PDF Loader", options=converter_options, key=converter_name
        )

        # Splitter options with tooltips
        st.selectbox(
            label="Split by",
            options=["word", "sentence", "passage", "page"],
            index=0,
            help="Choose the unit for splitting the document: word, sentence, passage, or page.",
            key=split_by_name,
        )
        st.slider(
            label="Split Length",
            min_value=10,
            max_value=500,
            value=50,
            step=5,
            help="Specify the chunk size, which is the number of words, sentences, or passages.",
            key=split_leng_name,
        )
        st.slider(
            label="Split Overlap",
            min_value=0,
            max_value=100,
            value=10,
            step=5,
            help="Set the number of overlapping units (words, sentences, or passages) between chunks.",
            key=split_overlap_name,
        )
        st.slider(
            label="Split Threshold",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="Set the minimum number of units that the document fragment should have. Below this threshold, it will be attached to the previous chunk.",
            key=split_thres_name,
        )

    region = st.columns([1, 1])
    with region[0]:
        st.button(
            label="Clean", type="secondary", key="clean_btn", use_container_width=True
        )

    with region[1]:
        st.button(
            label="Start", type="primary", key="start_btn", use_container_width=True
        )

    if session["clean_btn"]:
        st.rerun()

    if session["start_btn"]:
        converter, cleaner, splitter = get_conv_clean_splitter(
            session=session,
            converter_name=converter_name,
            split_by_name=split_by_name,
            split_leng_name=split_leng_name,
            split_overlap_name=split_overlap_name,
            split_thres_name=split_thres_name,
        )
        start_function(
            converter=converter,
            cleaner=cleaner,
            splitter=splitter,
            pdf_path_list=find_pdf(temp_dir),
            **kwargs,
        )
