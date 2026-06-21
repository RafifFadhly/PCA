"""
app.py — Aplikasi Streamlit: Deteksi Kemiripan Wajah & Kompresi Gambar dengan PCA
UAS Aljabar Linear

User cukup upload 2 foto → sistem otomatis membangun dataset sementara
via augmentasi → melatih PCA → membandingkan wajah.
Tidak perlu menyiapkan folder dataset.
"""

import streamlit as st
import numpy as np
import cv2
import os
import sys
import tempfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))
from pca_model import (
    get_face_crop_image,
    compare_faces_auto,
    augment_face,
)
from pca_1 import (
    compress_image_pca,
    reconstruct_with_k_components,
    calculate_psnr,
    calculate_ssim,
)

# ── Konfigurasi ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCA Vision App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] * { color: #f8fafc !important; }

.app-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%);
    border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
    text-align: center; box-shadow: 0 4px 24px rgba(0,0,0,0.15);
}
.app-header h1 { color: white; font-size: 2rem; font-weight: 700; margin: 0; }
.app-header p  { color: #c7d2fe; font-size: 1rem; margin: 8px 0 0; }

.metric-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-card .value { font-size: 2rem; font-weight: 700; color: #1e3a8a; }
.metric-card .label { font-size: 0.85rem; color: #64748b; margin-top: 4px; }

.badge-similar {
    display: inline-block; background: #dcfce7; color: #166534;
    border: 1.5px solid #86efac; padding: 10px 28px; border-radius: 99px;
    font-weight: 700; font-size: 1.2rem;
}
.badge-different {
    display: inline-block; background: #fee2e2; color: #991b1b;
    border: 1.5px solid #fca5a5; padding: 10px 28px; border-radius: 99px;
    font-weight: 700; font-size: 1.2rem;
}
.section-title {
    font-size: 1.15rem; font-weight: 600; color: #1e293b;
    margin: 16px 0 10px; padding-bottom: 6px; border-bottom: 2px solid #e2e8f0;
}
.step-box {
    background: #f8fafc; border-left: 4px solid #7c3aed;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    font-size: 0.88rem; color: #334155; line-height: 1.9; margin: 8px 0;
}
.hline { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ── Helper ─────────────────────────────────────────────────────────────────────
def uploaded_to_array(f):
    return cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), cv2.IMREAD_COLOR)

def save_temp(f):
    suffix = os.path.splitext(f.name)[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(f.getvalue()); tmp.flush()
    return tmp.name

def sim_bar(value, threshold):
    pct = int(max(0, min(value, 1)) * 100)
    color = "#22c55e" if value >= threshold else "#ef4444"
    return f"""
    <div style="margin:12px 0">
      <div style="display:flex;justify-content:space-between;
                  font-size:0.85rem;color:#64748b;margin-bottom:4px">
        <span>Cosine Similarity</span><span><b>{value:.4f}</b></span>
      </div>
      <div style="background:#e2e8f0;border-radius:99px;height:12px">
        <div style="width:{pct}%;background:{color};border-radius:99px;height:12px;
                    transition:width .4s ease"></div>
      </div>
      <div style="display:flex;justify-content:space-between;
                  font-size:0.75rem;color:#94a3b8;margin-top:3px">
        <span>0</span><span>Threshold: {threshold:.2f}</span><span>1.0</span>
      </div>
    </div>"""


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 PCA Vision App")
    st.markdown("*UAS Aljabar Linear*")
    st.markdown("---")
    menu = st.radio(
        "Pilih Fitur",
        ["🏠 Beranda", "👤 Deteksi Kemiripan Wajah", "🗜️ Kompresi Gambar"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    **📌 Cara Pakai:**
    - Pilih fitur di atas
    - Upload foto langsung
    - Klik tombol analisis
    - Hasil langsung tampil ✨
    """)
    st.caption("Streamlit · OpenCV · Scikit-learn")


# ══════════════════════════════════════════════════════════════════════════════
# BERANDA
# ══════════════════════════════════════════════════════════════════════════════
if menu == "🏠 Beranda":
    st.markdown("""
    <div class="app-header">
        <h1>🎯 PCA Vision App</h1>
        <p>Implementasi Eigenfaces (PCA/SVD) pada Pengolahan Gambar &nbsp;|&nbsp; UAS Aljabar Linear</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);
                    border-radius:14px;padding:24px;min-height:200px">
            <div style="font-size:2.5rem">👤</div>
            <h3 style="color:#1e3a8a;margin:8px 0">Deteksi Kemiripan Wajah</h3>
            <p style="color:#1e40af;font-size:0.9rem">
                Upload dua foto wajah — sistem otomatis membangun dataset via augmentasi,
                melatih PCA/Eigenfaces, lalu membandingkan dengan Cosine Similarity.
                <b>Tidak perlu menyiapkan folder dataset.</b>
            </p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#f5f3ff,#ede9fe);
                    border-radius:14px;padding:24px;min-height:200px">
            <div style="font-size:2.5rem">🗜️</div>
            <h3 style="color:#4c1d95;margin:8px 0">Kompresi Gambar PCA</h3>
            <p style="color:#5b21b6;font-size:0.9rem">
                Upload satu gambar — lihat bagaimana PCA mengompresi gambar dengan
                mereduksi komponen utama dan bandingkan kualitasnya (PSNR/SSIM).
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📖 Alur Sistem (sesuai dokumen UAS)")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("**Fase Membangun Dataset (Otomatis)**")
        st.markdown("""<div class="step-box">
        1️⃣ Deteksi & crop wajah dari 2 foto (Haar Cascade)<br>
        2️⃣ Grayscale → Resize 100×100 → /255 → Flatten<br>
        3️⃣ Augmentasi: flip, brightness, rotasi, noise<br>
        &nbsp;&nbsp;&nbsp;&nbsp;→ ~62 sampel dari 2 foto<br>
        4️⃣ Bentuk matriks <b>X ∈ ℝ<sup>62×10000</sup></b><br>
        5️⃣ Centering: <b>Xc = X − mean_face</b><br>
        6️⃣ SVD: <b>Xc = U Σ Vᵀ</b> → Eigenfaces (Vk)
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("**Fase Perbandingan Wajah**")
        st.markdown("""<div class="step-box">
        1️⃣ Ambil vektor wajah <b>asli</b> (bukan augmentasi)<br>
        2️⃣ Centering: <b>face_c = face − mean_face</b><br>
        3️⃣ Proyeksi: <b>z = face_c · Vk</b><br>
        4️⃣ Hitung: <b>Cosine Similarity(z₁, z₂)</b><br>
        5️⃣ Keputusan: similarity ≥ threshold<br>
        &nbsp;&nbsp;&nbsp;&nbsp;→ <b>Mirip / Tidak Mirip</b>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DETEKSI KEMIRIPAN WAJAH
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "👤 Deteksi Kemiripan Wajah":
    st.markdown("""
    <div class="app-header">
        <h1>👤 Deteksi Kemiripan Wajah</h1>
        <p>Upload dua foto → analisis otomatis dengan PCA Eigenfaces + Cosine Similarity</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Pengaturan ─────────────────────────────────────────────────────────────
    with st.expander("⚙️ Pengaturan (opsional)", expanded=False):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            threshold = st.slider(
                "Threshold Cosine Similarity",
                min_value=0.50, max_value=0.99, value=0.80, step=0.01,
                help="Batas minimum similarity untuk dinyatakan 'Mirip'"
            )
            n_components = st.slider(
                "Jumlah Komponen PCA (k)",
                min_value=5, max_value=60, value=30, step=5,
                help="Jumlah eigenfaces. Dibatasi otomatis sesuai jumlah sampel augmentasi."
            )
        with col_s2:
            n_augment = st.slider(
                "Jumlah Augmentasi per Foto",
                min_value=10, max_value=60, value=30, step=5,
                help="Lebih banyak augmentasi = PCA lebih stabil. Total sampel = 2×(n+1)."
            )
            st.markdown(f"""
            <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;
                        padding:10px 14px;font-size:0.85rem;color:#0c4a6e;margin-top:8px">
                📊 Total sampel dataset: <b>{2*(n_augment+1)}</b><br>
                🧩 Max komponen PCA: <b>{min(n_components, 2*(n_augment+1)-1)}</b><br>
                ✅ Tidak perlu folder dataset
            </div>""", unsafe_allow_html=True)

    # ── Upload foto ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📤 Upload Dua Foto Wajah</div>',
                unsafe_allow_html=True)

    col_u1, col_u2 = st.columns(2, gap="large")
    with col_u1:
        st.markdown("**Foto Wajah 1**")
        file1 = st.file_uploader("foto1", type=["jpg","jpeg","png","webp"],
                                 key="f1", label_visibility="collapsed")
    with col_u2:
        st.markdown("**Foto Wajah 2**")
        file2 = st.file_uploader("foto2", type=["jpg","jpeg","png","webp"],
                                 key="f2", label_visibility="collapsed")

    # Preview
    if file1 or file2:
        p1, p2 = st.columns(2, gap="large")
        with p1:
            if file1:
                a = uploaded_to_array(file1); file1.seek(0)
                st.image(cv2.cvtColor(a, cv2.COLOR_BGR2RGB),
                         caption="Foto 1", use_container_width=True)
        with p2:
            if file2:
                a = uploaded_to_array(file2); file2.seek(0)
                st.image(cv2.cvtColor(a, cv2.COLOR_BGR2RGB),
                         caption="Foto 2", use_container_width=True)

    # ── Tombol analisis ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    btn = st.button("🔍 Analisis Kemiripan", type="primary", use_container_width=True)

    if btn:
        if not file1 or not file2:
            st.error("❌ Upload kedua foto wajah terlebih dahulu.")
        else:
            file1.seek(0); file2.seek(0)
            tmp1 = save_temp(file1)
            tmp2 = save_temp(file2)

            try:
                # Preview crop wajah
                crop1 = get_face_crop_image(tmp1)
                crop2 = get_face_crop_image(tmp2)

                # ── Proses utama ───────────────────────────────────────────────
                with st.spinner("Membangun dataset augmentasi & melatih PCA..."):
                    similarity, result, pca_model, fv1, fv2 = compare_faces_auto(
                        tmp1, tmp2,
                        n_components=n_components,
                        n_augment=n_augment,
                        threshold=threshold
                    )

                st.markdown("<hr class='hline'>", unsafe_allow_html=True)
                st.markdown('<div class="section-title">📊 Hasil Analisis</div>',
                            unsafe_allow_html=True)

                if similarity is None:
                    st.error(f"Gagal memproses: {result}")
                    st.stop()

                # ── Wajah terdeteksi ───────────────────────────────────────────
                st.markdown("**Wajah yang Terdeteksi & Di-preprocess (100×100 px):**")
                cc1, cc2, cc3 = st.columns([1, 1, 2])
                with cc1:
                    if crop1 is not None:
                        st.image(crop1, caption="Wajah 1 (crop)", width=155)
                    else:
                        st.warning("Wajah 1 tidak terdeteksi")
                with cc2:
                    if crop2 is not None:
                        st.image(crop2, caption="Wajah 2 (crop)", width=155)
                    else:
                        st.warning("Wajah 2 tidak terdeteksi")
                with cc3:
                    n_actual = pca_model.n_components_
                    total_samples = 2 * (n_augment + 1)
                    explained = float(np.sum(pca_model.explained_variance_ratio_)) * 100
                    st.markdown(f"""
                    <div style="padding:4px 0;font-size:0.88rem;color:#475569">
                        <b>Detail proses:</b><br>
                        • Augmentasi per foto: {n_augment} variasi<br>
                        • Total sampel dataset: {total_samples}<br>
                        • Komponen PCA (k): {n_actual}<br>
                        • Varians explained: {explained:.1f}%<br>
                        • Threshold: {threshold}
                    </div>""", unsafe_allow_html=True)

                # ── Metric cards ───────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="value">{similarity:.4f}</div>
                        <div class="label">Cosine Similarity</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="value">{threshold:.2f}</div>
                        <div class="label">Threshold</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    pct = int(max(0, similarity) * 100)
                    st.markdown(f"""<div class="metric-card">
                        <div class="value">{pct}%</div>
                        <div class="label">Kemiripan</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(sim_bar(similarity, threshold), unsafe_allow_html=True)

                # Badge hasil
                is_sim = similarity >= threshold
                badge = "badge-similar" if is_sim else "badge-different"
                st.markdown(f"""
                <div style="text-align:center;margin:20px 0">
                    <span class="{badge}">{result}</span>
                </div>""", unsafe_allow_html=True)

                if is_sim:
                    st.success(
                        f"✅ Wajah **Mirip** — similarity **{similarity:.4f}** "
                        f"≥ threshold **{threshold:.2f}**"
                    )
                else:
                    st.info(
                        f"ℹ️ Wajah **Tidak Mirip** — similarity **{similarity:.4f}** "
                        f"< threshold **{threshold:.2f}**"
                    )

                # ── Visualisasi Eigenfaces ─────────────────────────────────────
                with st.expander("🧠 Visualisasi Eigenfaces"):
                    st.markdown("""
                    Eigenfaces hasil SVD dari dataset augmentasi — setiap wajah direpresentasikan
                    sebagai kombinasi linear dari komponen-komponen ini.
                    """)
                    n_show = min(10, pca_model.n_components_)
                    rows = 2 if n_show > 5 else 1
                    cols_ef = 5 if n_show > 5 else n_show
                    fig, axes = plt.subplots(rows, cols_ef, figsize=(cols_ef * 2.2, rows * 2.4))
                    axes = np.array(axes).flatten()
                    for i in range(n_show):
                        ef = pca_model.components_[i].reshape(100, 100)
                        axes[i].imshow(ef, cmap='gray')
                        axes[i].set_title(f"EF {i+1}", fontsize=9)
                        axes[i].axis('off')
                    for j in range(n_show, len(axes)):
                        axes[j].axis('off')
                    plt.suptitle(f"{n_show} Eigenfaces Pertama (dari {pca_model.n_components_} total)",
                                 fontsize=10, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # ── Mean Face ──────────────────────────────────────────────────
                with st.expander("👤 Mean Face (Wajah Rata-rata Dataset Augmentasi)"):
                    mean_face = pca_model.mean_.reshape(100, 100)
                    mean_disp = np.clip(mean_face * 255, 0, 255).astype(np.uint8)
                    c_mf1, c_mf2 = st.columns([1, 3])
                    with c_mf1:
                        st.image(mean_disp, caption="Mean Face", width=140)
                    with c_mf2:
                        st.markdown(f"""
                        Mean face adalah rata-rata dari **{total_samples} sampel augmentasi**.
                        Setiap wajah uji dikurangi mean face ini sebelum diproyeksikan
                        ke ruang PCA **(centering: Xc = X − mean_face)**.
                        """)

                # ── Contoh Augmentasi ──────────────────────────────────────────
                with st.expander("🔄 Contoh Augmentasi yang Dibuat"):
                    st.markdown("Beberapa contoh augmentasi dari **Foto 1** yang digunakan sebagai dataset:")
                    aug_samples = augment_face(fv1, n_augment=8)
                    aug_cols = st.columns(min(8, len(aug_samples)))
                    for i, (col, vec) in enumerate(zip(aug_cols, aug_samples[:8])):
                        img_aug = np.clip(vec.reshape(100, 100) * 255, 0, 255).astype(np.uint8)
                        with col:
                            st.image(img_aug,
                                     caption="Asli" if i == 0 else f"Aug {i}",
                                     use_container_width=True)

            finally:
                for t in [tmp1, tmp2]:
                    try: os.unlink(t)
                    except: pass


# ══════════════════════════════════════════════════════════════════════════════
# KOMPRESI GAMBAR
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🗜️ Kompresi Gambar":
    st.markdown("""
    <div class="app-header">
        <h1>🗜️ Kompresi Gambar dengan PCA</h1>
        <p>Reduksi dimensi gambar menggunakan SVD — lihat efek jumlah komponen terhadap kualitas</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📤 Upload Gambar</div>', unsafe_allow_html=True)
    img_file = st.file_uploader(
        "Pilih gambar", type=["jpg","jpeg","png","webp","bmp"],
        key="cimg", label_visibility="collapsed"
    )

    if img_file:
        img_arr = uploaded_to_array(img_file)
        gray_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        H, W = gray_img.shape

        ci1, ci2 = st.columns([1, 2])
        with ci1:
            st.image(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB),
                     caption=f"Asli ({W}×{H} px)", use_container_width=True)
        with ci2:
            max_k = min(H, W)
            st.markdown(f"""
            <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;
                        padding:14px 18px;color:#0c4a6e;font-size:0.9rem">
                <b>📋 Info Gambar</b><br><br>
                🖼️ Ukuran       : {W} × {H} px<br>
                📊 Total piksel : {W*H:,}<br>
                ⚙️ Max komponen : {max_k}
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            n_comp = st.slider("🔢 Jumlah Komponen PCA (k)",
                               1, min(max_k, 200), min(50, max_k), 1)
            show_multi = st.checkbox("Tampilkan perbandingan multi-level", value=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗜️ Mulai Kompresi", type="primary", use_container_width=True):
            with st.spinner(f"Menjalankan PCA dengan k={n_comp}..."):
                res = compress_image_pca(gray_img, n_components=n_comp)

            st.markdown("<hr class='hline'>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Hasil Kompresi</div>', unsafe_allow_html=True)

            mc1, mc2, mc3, mc4 = st.columns(4)
            for col, (val, lbl) in zip([mc1,mc2,mc3,mc4], [
                (f"{n_comp}/{res['max_components']}", "Komponen PCA"),
                (f"{res['explained_var']:.1f}%", "Varians Explained"),
                (f"{res['psnr']:.1f} dB", "PSNR"),
                (f"{max(0,res['compression_ratio']):.1f}%", "Rasio Kompresi"),
            ]):
                with col:
                    st.markdown(f"""<div class="metric-card">
                        <div class="value">{val}</div>
                        <div class="label">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            co1, co2 = st.columns(2, gap="large")
            with co1:
                st.markdown("**🖼️ Gambar Asli**")
                st.image(res['original'], use_container_width=True)
            with co2:
                st.markdown(f"**🔧 Rekonstruksi (k={n_comp})**")
                st.image(res['reconstructed'],
                         caption=f"PSNR: {res['psnr']} dB", use_container_width=True)

            p = res['psnr']
            if p == float('inf'): st.success("🌟 Rekonstruksi sempurna")
            elif p > 40: st.success(f"✅ Kualitas sangat baik (PSNR {p:.1f} dB)")
            elif p > 30: st.warning(f"⚠️ Kualitas baik (PSNR {p:.1f} dB)")
            else:        st.error(f"❌ Kualitas rendah (PSNR {p:.1f} dB)")

            if show_multi:
                st.markdown("<hr class='hline'>", unsafe_allow_html=True)
                st.markdown('<div class="section-title">🔢 Perbandingan Multi-Level</div>',
                            unsafe_allow_html=True)
                with st.spinner("Menghitung berbagai level..."):
                    k_list = sorted(set([1,5,10,20,50,n_comp,
                                         min(100,res['max_components']),
                                         res['max_components']]))
                    k_list = [k for k in k_list if 0 < k <= res['max_components']]
                    multi = reconstruct_with_k_components(gray_img, k_list=k_list)

                for i in range(0, len(multi), 4):
                    rcols = st.columns(4)
                    for j, r in enumerate(multi[i:i+4]):
                        with rcols[j]:
                            st.image(r['image'],
                                     caption=f"k={r['k']} | {r['psnr']}dB | {r['explained_var']:.0f}%",
                                     use_container_width=True)

                # Grafik
                ks    = [r['k']    for r in multi if r['psnr'] != float('inf')]
                psnrs = [r['psnr'] for r in multi if r['psnr'] != float('inf')]
                vars_ = [r['explained_var'] for r in multi if r['psnr'] != float('inf')]
                if ks:
                    fig, ax1 = plt.subplots(figsize=(9, 4))
                    ax1.plot(ks, psnrs, 'o-', color='#1e3a8a', lw=2, ms=6, label='PSNR (dB)')
                    ax1.axhline(30, color='#ef4444', ls='--', alpha=.6, label='30 dB')
                    ax1.axhline(40, color='#22c55e', ls='--', alpha=.6, label='40 dB')
                    if n_comp in ks:
                        ax1.axvline(n_comp, color='#7c3aed', ls=':', alpha=.8, label=f'k={n_comp}')
                    ax1.set_xlabel('k (Komponen PCA)', fontsize=11)
                    ax1.set_ylabel('PSNR (dB)', color='#1e3a8a', fontsize=11)
                    ax2 = ax1.twinx()
                    ax2.plot(ks, vars_, 's--', color='#f59e0b', lw=1.5, ms=5, alpha=.7,
                             label='Varians (%)')
                    ax2.set_ylabel('Varians (%)', color='#f59e0b', fontsize=11)
                    lines = ax1.get_legend_handles_labels()[0]+ax2.get_legend_handles_labels()[0]
                    lbls  = ax1.get_legend_handles_labels()[1]+ax2.get_legend_handles_labels()[1]
                    ax1.legend(lines, lbls, loc='lower right', fontsize=9)
                    ax1.set_title(f'PSNR vs Jumlah Komponen PCA ({W}×{H} px)',
                                  fontsize=12, fontweight='bold')
                    ax1.grid(True, alpha=.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            with st.expander("🔧 Detail Teknis"):
                ssim_val = calculate_ssim(res['original'], res['reconstructed'])
                comp_size = (H*n_comp)+(n_comp*W)+n_comp
                st.markdown(f"""
                | Parameter | Nilai |
                |-----------|-------|
                | Ukuran gambar | {W}×{H} px |
                | Total piksel asli | {H*W:,} |
                | Komponen PCA (k) | {n_comp} |
                | Data terkompresi | {comp_size:,} nilai |
                | Rasio kompresi | {max(0,res['compression_ratio']):.2f}% |
                | Varians explained | {res['explained_var']:.2f}% |
                | PSNR | {res['psnr']} dB |
                | SSIM | {ssim_val:.4f} |
                """)
