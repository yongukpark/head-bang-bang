import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_OPTIONS = [
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
]
DEFAULT_MODEL_NAME = "EleutherAI/pythia-1.4b"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_selected_model_name() -> str:
    """Render global model selector and persist it in Streamlit session state."""
    if "selected_model_name" not in st.session_state:
        st.session_state["selected_model_name"] = DEFAULT_MODEL_NAME

    if st.session_state["selected_model_name"] not in MODEL_OPTIONS:
        st.session_state["selected_model_name"] = DEFAULT_MODEL_NAME

    return st.sidebar.selectbox(
        "Model (All Pages)",
        options=MODEL_OPTIONS,
        key="selected_model_name",
    )


@st.cache_resource
def load_model(model_name: str, device_name: str):
    """Load and cache model/tokenizer pair per (model_name, device)."""
    device = torch.device(device_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer
