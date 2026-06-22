import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image

# Import modul PCA buatanmu
import pca_model
import pca_1

# ======================================
# KONFIGURASI HALAMAN
# ======================================
st.set_page_config(
    page_title="Aplikasi PCA Multiguna",
    layout="centered"
)

# ======================================
# NAVIGASI SIDEBAR
# ======================================
st.sidebar.title("Navigasi Fitur")
menu = st.sidebar.selectbox(
    "Pilih Fitur PCA:",
    ["Cek Kemiripan Wajah", "Kompresi Gambar"]
)

# Folder temporary untuk menyimpan file upload-an sementara
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ======================================
# FITUR 1: CEK KEMIRIPAN WAJAH
# ======================================
if menu == "Cek Kemiripan Wajah":
    st.title("Deteksi Kemiripan Wajah Menggunakan PCA")
    st.write("Fitur ini membandingkan dua wajah berdasarkan ruang dimensi rendah PCA.")

    # LOAD MODEL DATASET (HANYA SEKALI)
    DATASET_PATH = "dataset"
    
    # Cek apakah folder dataset ada sebelum inisialisasi
    if not os.path.exists(DATASET_PATH):
        st.warning(f"Folder '{DATASET_PATH}' tidak ditemukan. Buat folder tersebut dan isi dengan sub-folder berisi foto wajah untuk training PCA wajah.")
    else:
        if "model_loaded" not in st.session_state:
            with st.spinner("Memuat model PCA wajah..."):
                try:
                    pca_model.initialize_model(DATASET_PATH)
                    st.session_state.model_loaded = True
                except Exception as e:
                    st.error(f"Gagal memuat model: {e}")

    # UPLOAD GAMBAR WAJAH
    uploaded_file1 = st.file_uploader("Upload Gambar 1", type=["jpg", "jpeg", "png"], key="face1")
    uploaded_file2 = st.file_uploader("Upload Gambar 2", type=["jpg", "jpeg", "png"], key="face2")

    if uploaded_file1 and uploaded_file2:
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open(uploaded_file1), caption="Gambar 1", use_container_width=True)
        with col2:
            st.image(Image.open(uploaded_file2), caption="Gambar 2", use_container_width=True)

        threshold = st.slider("Threshold Kemiripan", min_value=0.0, max_value=1.0, value=0.75, step=0.05)

        if st.button("Bandingkan Wajah"):
            if "model_loaded" not in st.session_state:
                st.error("Model PCA belum siap karena dataset kosong atau tidak ditemukan.")
            else:
                path1 = os.path.join(TEMP_DIR, uploaded_file1.name)
                path2 = os.path.join(TEMP_DIR, uploaded_file2.name)

                with open(path1, "wb") as f:
                    f.write(uploaded_file1.getbuffer())
                with open(path2, "wb") as f:
                    f.write(uploaded_file2.getbuffer())

                with st.spinner("Menghitung kemiripan..."):
                    similarity, status = pca_model.compare_uploaded_images(
                        path1, path2, threshold=threshold
                    )

                st.divider()
                st.subheader("Hasil Analisis")

                if similarity is not None:
                    # Menggunakan st.metric agar visualisasi hasil lebih menarik
                    st.metric(label="Persentase Kemiripan", value=f"{similarity*100:.2f}%")
                    if status == "Mirip":
                        st.success(f"Status: {status}")
                    else:
                        st.warning(f"Status: {status}")
                else:
                    st.error(status)

# ======================================
# FITUR 2: KOMPRESI GAMBAR
# ======================================
elif menu == "Kompresi Gambar":
    st.title("Kompresi Gambar Menggunakan PCA")
    st.write("Fitur ini mengurangi dimensi piksel gambar menggunakan komponen utama PCA.")

    uploaded_img_comp = st.file_uploader("Upload Gambar untuk Dikompresi", type=["jpg", "jpeg", "png"], key="comp_img")

    if uploaded_img_comp:
        # Simpan gambar sementara
        path_comp = os.path.join(TEMP_DIR, uploaded_img_comp.name)
        with open(path_comp, "wb") as f:
            f.write(uploaded_img_comp.getbuffer())

        # Baca dimensi gambar asli untuk membatasi slider n_components
        try:
            gray_temp = pca_1.load_image_grayscale(path_comp)
            max_k = min(gray_temp.shape[0], gray_temp.shape[1])
        except Exception as e:
            st.error(f"Error membaca gambar: {e}")
            max_k = 100

        # Input Jumlah Komponen Utama (n_components)
        n_components = st.slider(
            "Jumlah Komponen PCA (n_components)", 
            min_value=1, 
            max_value=max_k, 
            value=min(50, max_k),
            help="Semakin kecil nilai komponen, gambar semakin terkompresi namun kualitas berkurang."
        )

        if st.button("Proses Kompresi"):
            with st.spinner("Mengompresi gambar via PCA..."):
                res = pca_1.compress_image_pca(path_comp, n_components=n_components)

            st.divider()
            st.subheader("Perbandingan Gambar")

            # Tampilkan gambar berdampingan (Asli vs Rekonstruksi)
            col1, col2 = st.columns(2)
            with col1:
                st.image(res['original'], caption="Gambar Asli (Grayscale)", use_container_width=True)
            with col2:
                st.image(res['reconstructed'], caption=f"Hasil Rekonstruksi ({res['n_components']} Komponen)", use_container_width=True)

            # Tampilkan Metrik Kualitas Kompresi
            st.subheader("Metrik Kualitas")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Varians Dijelaskan", f"{res['explained_var']:.2f}%")
            m_col2.metric("Rasio Kompresi", f"{res['compression_ratio']:.2f}%")
            m_col3.metric("Kualitas (PSNR)", f"{res['psnr']} dB")

            # Evaluasi Nilai PSNR
            if res['psnr'] >= 40:
                st.success("Kualitas Rekonstruksi Sangat Baik!")
            elif res['psnr'] >= 30:
                st.info("Kualitas Rekonstruksi Baik.")
            else:
                st.warning("Kualitas Rekonstruksi Cukup/Rendah (Terlihat distorsi).")
