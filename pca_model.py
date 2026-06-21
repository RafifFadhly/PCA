import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Ukuran standar gambar wajah sesuai dokumen
IMG_SIZE = (100, 100)

# ==========================================
# 1. FUNGSI DETEKSI DAN PREPROCESSING
# ==========================================
def detect_and_crop_face(image_path):
    """
    Mendeteksi wajah dari gambar menggunakan Haar Cascade, 
    lalu mengembalikan crop wajah yang sudah di-preprocess.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gambar tidak ditemukan: {image_path}")

    # Ubah ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Inisialisasi Haar Cascade untuk deteksi wajah frontal 
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) == 0:
        raise ValueError(f"Wajah tidak terdeteksi pada gambar: {image_path}")

    # Ambil wajah pertama yang terdeteksi 
    x, y, w, h = faces[0]
    
    # Crop area wajah [cite: 260]
    face_crop = gray[y:y+h, x:x+w]
    
    # Resize ke ukuran standar 100x100 [cite: 260]
    face_resized = cv2.resize(face_crop, IMG_SIZE)
    
    # Normalisasi nilai piksel ke rentang 0-1 [cite: 260]
    face_normalized = face_resized / 255.0
    
    # Flatten menjadi vektor 1D [cite: 260]
    return face_normalized.flatten()

def get_face_crop_image(image_path):
    """
    Mengembalikan gambar wajah hasil crop (100x100)
    untuk ditampilkan di Streamlit.
    """
    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    crop = gray[y:y+h, x:x+w]
    crop = cv2.resize(crop, IMG_SIZE)

    return crop

# ==========================================
# 2. PERSIAPAN DATA LATIH (DATASET)
# ==========================================
def load_dataset(dataset_path):
    """
    Membaca seluruh gambar wajah dari folder dataset[cite: 161, 163].
    """
    X = []
    labels = []
    
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
            
        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(person_folder, filename)
                try:
                    # Menggunakan fungsi deteksi wajah baru
                    vector = detect_and_crop_face(image_path)
                    X.append(vector)
                    labels.append(person_name)
                except ValueError as e:
                    print(f"Melewati {filename}: {e}")
                    
    return np.array(X), np.array(labels)

def augment_face(face_vector, n_augment=30):
    """
    Membuat beberapa variasi wajah sebagai dataset augmentasi.
    """
    img = face_vector.reshape(IMG_SIZE)

    samples = [face_vector]

    rng = np.random.default_rng(42)

    while len(samples) < n_augment + 1:

        aug = img.copy()

        mode = rng.integers(4)

        if mode == 0:
            # Flip horizontal
            aug = cv2.flip(aug, 1)

        elif mode == 1:
            # Brightness
            alpha = rng.uniform(0.8, 1.2)
            aug = np.clip(aug * alpha, 0, 1)

        elif mode == 2:
            # Rotasi
            angle = rng.uniform(-10, 10)

            M = cv2.getRotationMatrix2D((50, 50), angle, 1)

            aug = cv2.warpAffine(
                aug,
                M,
                IMG_SIZE,
                borderMode=cv2.BORDER_REFLECT
            )

        else:
            # Noise
            noise = rng.normal(0, 0.02, IMG_SIZE)
            aug = np.clip(aug + noise, 0, 1)

        samples.append(aug.flatten())

    return samples

def train_pca(X, n_components=30):
    """
    Melatih model PCA dari dataset wajah.
    """
    n_components = min(
        n_components,
        len(X) - 1,
        X.shape[1]
    )

    pca = PCA(n_components=n_components)

    pca.fit(X)

    return pca

# ==========================================
# 3. FUNGSI PERBANDINGAN WAJAH (DATA UJI)
# ==========================================
def compare_faces(image_path_1, image_path_2, pca_model, threshold=0.80):
    """
    Membandingkan dua gambar wajah menggunakan PCA dan cosine similarity[cite: 191].
    """
    try:
        # Preprocessing Data Uji
        face_1 = detect_and_crop_face(image_path_1)
        face_2 = detect_and_crop_face(image_path_2)
    except ValueError as e:
        return None, str(e)

    # Ubah menjadi bentuk 2D karena PCA membutuhkan input 2D [cite: 194]
    face_1 = face_1.reshape(1, -1)
    face_2 = face_2.reshape(1, -1)

    # Proyeksi ke ruang PCA [cite: 197, 198, 199]
    face_1_pca = pca_model.transform(face_1)
    face_2_pca = pca_model.transform(face_2)

    # Hitung cosine similarity [cite: 200, 201]
    similarity = cosine_similarity(face_1_pca, face_2_pca)[0][0]

    # Keputusan Kemiripan
    if similarity >= threshold:
        result = "Mirip"
    else:
        result = "Tidak mirip"

    return similarity, result

def compare_faces_auto(
    image_path_1,
    image_path_2,
    n_components=30,
    n_augment=30,
    threshold=0.80
):
    """
    Versi otomatis tanpa folder dataset.
    Dataset dibuat dari augmentasi dua foto yang di-upload.
    """

    try:
        # Preprocessing kedua wajah
        face1 = detect_and_crop_face(image_path_1)
        face2 = detect_and_crop_face(image_path_2)

    except ValueError as e:
        return None, str(e), None, None, None

    # Bangun dataset sementara
    dataset = []

    dataset.extend(
        augment_face(face1, n_augment)
    )

    dataset.extend(
        augment_face(face2, n_augment)
    )

    X = np.array(dataset)

    # Latih PCA
    pca = train_pca(
        X,
        n_components=n_components
    )

    # Bandingkan kedua wajah
    similarity, result = compare_faces(
        image_path_1,
        image_path_2,
        pca,
        threshold=threshold
    )

    return (
        similarity,
        result,
        pca,
        face1,
        face2
    )

# ==========================================
# 4. ALUR EKSEKUSI UTAMA (MAIN)
# ==========================================
if __name__ == "__main__":
    # --- TAHAP 1: DATA LATIH ---
    # Pastikan kamu memiliki folder 'dataset' yang berisi subfolder nama orang [cite: 23, 24, 25]
    folder_dataset = r"D:\\TEKNIK_INFORMATIKA\SEMESTER 2\ALJABAR LINEAR\\UAS\dataset" 
    
    print("Memuat dataset dan mengekstraksi wajah...")
    X_train, labels = load_dataset(folder_dataset)
    
    if len(X_train) == 0:
        print(f"Error: Tidak ada wajah yang berhasil diekstrak dari folder '{folder_dataset}'.")
        exit()

    print(f"Berhasil memuat {len(X_train)} gambar wajah.")

    # Latih PCA [cite: 177, 178]
    # Sesuai dokumen, kita gunakan 50 komponen utama (atau maksimal jumlah sampel)
    n_components = min(50, len(X_train))
    pca = PCA(n_components=n_components)
    
    # Melakukan PCA pada data latih [cite: 179]
    X_pca = pca.fit_transform(X_train)
    print(f"Model PCA berhasil dilatih. Total varians: {np.sum(pca.explained_variance_ratio_):.4f}")

    # --- TAHAP 2: DATA UJI ---
    print("\nMembandingkan wajah...")
    path_wajah_1 = r"D:\\TEKNIK_INFORMATIKA\SEMESTER 2\ALJABAR LINEAR\\UAS\\s3.jpeg"  # Ganti dengan path foto pertama
    path_wajah_2 = r"D:\\TEKNIK_INFORMATIKA\SEMESTER 2\ALJABAR LINEAR\\UAS\\s3.jpeg"  # Ganti dengan path foto kedua
    
    # Gunakan threshold 0.75 sesuai contoh di dokumen [cite: 137, 211]
    THRESHOLD_VALUE = 0.75 
    
    similarity_score, status = compare_faces(path_wajah_1, path_wajah_2, pca, threshold=THRESHOLD_VALUE)
    
    # --- TAMPILKAN HASIL ---
    print("\n" + "="*40)
    print("HASIL DETEKSI KEMIRIPAN WAJAH")
    print("="*40)
    if similarity_score is not None:
        print(f"Gambar 1       : {path_wajah_1}")
        print(f"Gambar 2       : {path_wajah_2}")
        print(f"Threshold      : {THRESHOLD_VALUE}")
        print(f"Nilai Cosine   : {similarity_score:.4f}")
        print(f"Status         : {status}")
    else:
        print(f"Gagal memproses: {status}")
    print("="*40)