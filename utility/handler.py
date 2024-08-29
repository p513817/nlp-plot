import random
from typing import Dict, Union

import streamlit as st
import torch
import torch.nn.functional as F
from haystack.components.converters import PDFMinerToDocument, PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from transformers import AutoModel, AutoTokenizer

PDFMiner = "PDFMiner"
PyPDF = "PyPDF"
SUP_CONVS: list = [PyPDF, PDFMiner]
FUNC_CONVS: dict = {PyPDF: PyPDFToDocument, PDFMiner: PDFMinerToDocument}


# Function to generate a random color with alpha transparency
def generate_random_color(alpha=0.5):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"rgba({r}, {g}, {b}, {alpha})"


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# PCA using SVD in PyTorch
def pca_svd(X, k=2):
    # Center the data
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean
    # Perform SVD
    U, S, V = torch.svd(X_centered)
    # Select the top-k components
    X_pca = torch.mm(X_centered, V[:, :k])

    return X_pca


@st.cache_resource
def get_tokenizer_and_model() -> tuple:
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    print("Loaded Tokenizer and Model")
    return tokenizer, model


def sentence_to_embedding(sentences: list, tokenizer_model=get_tokenizer_and_model()):
    tokenizer, model = tokenizer_model

    # Tokenize sentences and prompt
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


def sentence_embedding_to_2d(sentence_embeddings: list) -> list:
    # Perform PCA
    sentence_embeddings_2d = pca_svd(sentence_embeddings, k=2).numpy()
    return sentence_embeddings_2d


@st.cache_resource
def get_converters() -> Dict[str, Union[PyPDFToDocument, PDFMinerToDocument]]:
    return FUNC_CONVS


@st.cache_resource
def get_cleaner(
    remove_empty_lines: bool = True,
    remove_extra_whitespaces: bool = True,
    remove_repeated_substrings: bool = False,
):
    return DocumentCleaner(
        remove_empty_lines=remove_empty_lines,
        remove_extra_whitespaces=remove_extra_whitespaces,
        remove_repeated_substrings=remove_repeated_substrings,
    )


def get_splitter(split_by, split_length, split_overlap, split_threshold):
    return DocumentSplitter(
        split_by=split_by,
        split_length=split_length,
        split_overlap=split_overlap,
        split_threshold=split_threshold,
    )
