"""
pca_1.py — Kompresi Gambar Menggunakan PCA
Menggunakan PCA untuk mereduksi dimensi gambar grayscale,
kemudian merekonstruksi gambar dari komponen utama yang lebih sedikit.
"""

import cv2
import numpy as np
from sklearn.decomposition import PCA
import os


# ==========================================
# 1. LOAD DAN PREPROCESS GAMBAR
# ==========================================
def load_image_grayscale(image_input):
    """
    Membaca gambar dan mengubah ke grayscale.

    Args:
        image_input: path file (str) atau numpy array BGR/grayscale

    Returns:
        img_gray (np.ndarray): gambar grayscale uint8
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Gambar tidak ditemukan: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input
        if img.size == 0:
            raise ValueError("Array gambar kosong.")
    else:
        raise ValueError("Input harus berupa path file (str) atau numpy array.")

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    return gray


# ==========================================
# 2. KOMPRESI GAMBAR DENGAN PCA
# ==========================================
def compress_image_pca(image_input, n_components=50):
    """
    Mengompresi gambar menggunakan PCA.

    Cara kerja:
      - Gambar grayscale (H x W) diperlakukan sebagai matriks H baris,
        setiap baris adalah vektor intensitas piksel 1 x W.
      - PCA dilakukan pada baris-baris tersebut untuk mendapat
        representasi dalam ruang berdimensi lebih rendah.
      - Rekonstruksi dilakukan dengan inverse_transform.

    Args:
        image_input: path file (str) atau numpy array
        n_components (int): jumlah komponen PCA (makin kecil = makin terkompresi)

    Returns:
        dict berisi:
            'original'       : gambar asli grayscale (np.ndarray uint8)
            'reconstructed'  : gambar rekonstruksi (np.ndarray uint8)
            'n_components'   : komponen PCA yang digunakan
            'max_components' : komponen maksimal yang tersedia
            'compression_ratio': rasio kompresi (%)
            'psnr'           : Peak Signal-to-Noise Ratio (kualitas rekonstruksi)
            'explained_var'  : total varians yang dijelaskan (%)
            'pca'            : objek PCA yang sudah dilatih
    """
    # Load gambar
    gray = load_image_grayscale(image_input)
    H, W = gray.shape

    # Konversi ke float64 untuk PCA
    img_float = gray.astype(np.float64)

    # Batasi n_components sesuai dimensi gambar
    max_components = min(H, W)
    n_components = min(n_components, max_components)

    # Latih PCA pada baris-baris gambar
    pca = PCA(n_components=n_components, svd_solver='full')
    transformed = pca.fit_transform(img_float)   # shape: (H, n_components)

    # Rekonstruksi gambar dari komponen terbatas
    reconstructed_float = pca.inverse_transform(transformed)  # shape: (H, W)

    # Clip dan konversi kembali ke uint8
    reconstructed = np.clip(reconstructed_float, 0, 255).astype(np.uint8)

    # Hitung rasio kompresi
    # Ukuran asli: H * W piksel
    # Ukuran terkompresi: H * n_components (skor) + n_components * W (komponen) + n_components (mean)
    original_size = H * W
    compressed_size = (H * n_components) + (n_components * W) + n_components
    compression_ratio = (1 - compressed_size / original_size) * 100

    # Hitung PSNR (Peak Signal-to-Noise Ratio)
    psnr = calculate_psnr(gray, reconstructed)

    # Total varians yang dijelaskan
    explained_var = float(np.sum(pca.explained_variance_ratio_)) * 100

    return {
        'original': gray,
        'reconstructed': reconstructed,
        'n_components': n_components,
        'max_components': max_components,
        'compression_ratio': compression_ratio,
        'psnr': psnr,
        'explained_var': explained_var,
        'pca': pca,
        'image_shape': (H, W),
    }


# ==========================================
# 3. REKONSTRUKSI DENGAN KOMPONEN BERBEDA
# ==========================================
def reconstruct_with_k_components(gray_image, k_list=None):
    """
    Merekonstruksi gambar dengan berbagai jumlah komponen PCA
    untuk menunjukkan efek kompresi bertingkat.

    Args:
        gray_image (np.ndarray): gambar grayscale
        k_list (list[int]): daftar jumlah komponen yang ingin dicoba

    Returns:
        list of dict: setiap dict berisi 'k', 'image', 'psnr', 'explained_var'
    """
    H, W = gray_image.shape
    max_k = min(H, W)

    if k_list is None:
        # Default: coba berbagai tingkat kompresi
        k_list = [1, 5, 10, 20, 50, 100, 150, max_k]

    # Latih PCA sekali dengan komponen maksimal
    max_k_used = min(max(k_list), max_k)
    img_float = gray_image.astype(np.float64)
    pca_full = PCA(n_components=max_k_used, svd_solver='full')
    scores_full = pca_full.fit_transform(img_float)  # (H, max_k_used)

    results = []
    for k in k_list:
        k = min(k, max_k_used)
        if k <= 0:
            continue

        # Gunakan hanya k komponen pertama
        scores_k = scores_full[:, :k]
        components_k = pca_full.components_[:k, :]
        mean = pca_full.mean_

        reconstructed_float = scores_k @ components_k + mean
        reconstructed = np.clip(reconstructed_float, 0, 255).astype(np.uint8)

        psnr = calculate_psnr(gray_image, reconstructed)
        explained = float(np.sum(pca_full.explained_variance_ratio_[:k])) * 100

        results.append({
            'k': k,
            'image': reconstructed,
            'psnr': psnr,
            'explained_var': explained,
        })

    return results


# ==========================================
# 4. METRIK KUALITAS
# ==========================================
def calculate_psnr(original, reconstructed):
    """
    Menghitung Peak Signal-to-Noise Ratio (PSNR) dalam dB.
    Semakin tinggi PSNR, semakin baik kualitas rekonstruksi.
    - > 40 dB  : kualitas sangat baik
    - 30–40 dB : kualitas baik
    - < 30 dB  : kualitas cukup / terlihat perbedaan
    """
    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return round(float(psnr), 2)


def calculate_ssim(original, reconstructed):
    """
    Menghitung Structural Similarity Index (SSIM).
    Nilai mendekati 1 = sangat mirip dengan asli.
    """
    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)

    mu1 = np.mean(orig)
    mu2 = np.mean(recon)
    sigma1_sq = np.var(orig)
    sigma2_sq = np.var(recon)
    sigma12 = np.mean((orig - mu1) * (recon - mu2))

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return round(float(numerator / denominator), 4)


# ==========================================
# 5. ALUR EKSEKUSI UTAMA (CLI)
# ==========================================
if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("  KOMPRESI GAMBAR DENGAN PCA")
    print("=" * 50)

    if len(sys.argv) < 2:
        print("\nUsage: python pca_1.py <gambar> [n_components]")
        print("Contoh: python pca_1.py foto.jpg 50\n")
        sys.exit(0)

    image_path = sys.argv[1]
    n_comp = int(sys.argv[2]) if len(sys.argv) >= 3 else 50

    print(f"\nGambar       : {image_path}")
    print(f"N Components : {n_comp}")

    result = compress_image_pca(image_path, n_components=n_comp)

    print(f"\nUkuran Gambar      : {result['image_shape'][1]} x {result['image_shape'][0]} px")
    print(f"Komponen Digunakan : {result['n_components']} / {result['max_components']}")
    print(f"Varians Dijelaskan : {result['explained_var']:.2f}%")
    print(f"Rasio Kompresi     : {result['compression_ratio']:.2f}%")
    print(f"PSNR               : {result['psnr']} dB")

    # Simpan gambar rekonstruksi
    out_path = f"reconstructed_{n_comp}comp_{os.path.basename(image_path)}"
    cv2.imwrite(out_path, result['reconstructed'])
    print(f"\nHasil rekonstruksi disimpan ke: {out_path}")
    print("=" * 50)
