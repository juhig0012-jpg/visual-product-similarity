import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from search import VisualSearchEngine
from config import METADATA_CSV


st.set_page_config(page_title="Visual Product Similarity Search", layout="wide")
st.title("Visual Product Similarity & Recommendation System")

st.write(
    "Upload a product image to find visually similar items using deep learning embeddings and FAISS."
)

@st.cache_resource
def load_engine():
    return VisualSearchEngine()


@st.cache_data
def load_metadata():
    return pd.read_csv(METADATA_CSV) if Path(METADATA_CSV).exists() else pd.DataFrame()


engine = load_engine()
metadata_df = load_metadata()

categories = sorted(metadata_df["category"].dropna().unique().tolist()) if "category" in metadata_df.columns else []
availability_options = sorted(metadata_df["availability"].dropna().unique().tolist()) if "availability" in metadata_df.columns else []

with st.sidebar:
    st.header("Filters")
    selected_category = st.selectbox("Category", ["All"] + categories)
    min_price = st.number_input("Min Price", min_value=0.0, value=0.0, step=100.0)
    max_price = st.number_input("Max Price", min_value=0.0, value=100000.0, step=100.0)
    selected_availability = st.selectbox("Availability", ["All"] + availability_options)
    top_k = st.slider("Top K Results", min_value=1, max_value=20, value=5)

uploaded_file = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Query Image")
    st.image(query_image, width=300)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        query_image.save(tmp_file.name)
        query_image_path = tmp_file.name

    category_filter = None if selected_category == "All" else selected_category
    availability_filter = None if selected_availability == "All" else selected_availability

    results = engine.search(
        query_image_path=query_image_path,
        top_k=top_k,
        category=category_filter,
        min_price=min_price if min_price > 0 else None,
        max_price=max_price if max_price < 100000 else None,
        availability=availability_filter,
    )

    st.subheader("Similar Products")

    if not results:
        st.warning("No similar products found.")
    else:
        cols = st.columns(3)
        for idx, result in enumerate(results):
            col = cols[idx % 3]
            with col:
                st.image(result["image_path"], use_container_width=True)
                st.write(f"**Image:** {result.get('image_name', 'N/A')}")
                st.write(f"**Title:** {result.get('title', 'N/A')}")
                st.write(f"**Category:** {result.get('category', 'N/A')}")
                st.write(f"**Price:** {result.get('price', 'N/A')}")
                st.write(f"**Availability:** {result.get('availability', 'N/A')}")
                st.write(f"**Similarity Score:** {result.get('similarity_score', 0):.4f}")