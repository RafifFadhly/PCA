import streamlit as st
import os
from PIL import Image
import pca_model

# ======================================
# KONFIGURASI HALAMAN
# ======================================
st.set_page_config(
    page_title="Deteksi Kemiripan Wajah",
    layout="centered"
)

st.title("Deteksi Kemiripan Wajah Menggunakan PCA")

# ======================================
# LOAD MODEL (HANYA SEKALI)
# ======================================

DATASET_PATH = "dataset"

if "model_loaded" not in st.session_state:

    with st.spinner("Memuat model PCA..."):
        pca_model.initialize_model(DATASET_PATH)

    st.session_state.model_loaded = True

# ======================================
# UPLOAD GAMBAR
# ======================================

uploaded_file1 = st.file_uploader(
    "Upload Gambar 1",
    type=["jpg", "jpeg", "png"],
    key="img1"
)

uploaded_file2 = st.file_uploader(
    "Upload Gambar 2",
    type=["jpg", "jpeg", "png"],
    key="img2"
)

# ======================================
# PREVIEW
# ======================================

if uploaded_file1 and uploaded_file2:

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            Image.open(uploaded_file1),
            caption="Gambar 1",
            use_container_width=True
        )

    with col2:
        st.image(
            Image.open(uploaded_file2),
            caption="Gambar 2",
            use_container_width=True
        )

    # ======================================
    # TOMBOL PROSES
    # ======================================

    if st.button("Bandingkan"):

        os.makedirs("temp", exist_ok=True)

        path1 = os.path.join("temp", uploaded_file1.name)
        path2 = os.path.join("temp", uploaded_file2.name)

        with open(path1, "wb") as f:
            f.write(uploaded_file1.getbuffer())

        with open(path2, "wb") as f:
            f.write(uploaded_file2.getbuffer())

        with st.spinner("Menghitung kemiripan..."):

            similarity, status = pca_model.compare_uploaded_images(
                path1,
                path2,
                threshold=0.75
            )

        st.divider()

        st.subheader("Hasil")

        if similarity is not None:

            st.write(f"Persentase Kemiripan : {similarity*100:.2f}%")
            st.write(f"Status : {status}")

        else:

            st.error(status)
