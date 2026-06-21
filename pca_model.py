import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# KONFIGURASI
# =====================================================
IMG_SIZE = (100, 100)

# Variabel global model PCA
pca_model = None

# =====================================================
# 1. DETEKSI DAN PREPROCESSING WAJAH
# =====================================================
def detect_and_crop_face(image_path):
    """
    Mendeteksi wajah dari gambar menggunakan Haar Cascade,
    kemudian melakukan preprocessing.
    """

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Gambar tidak ditemukan: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) == 0:
        raise ValueError(f"Wajah tidak terdeteksi pada gambar: {image_path}")

    x, y, w, h = faces[0]

    face_crop = gray[y:y+h, x:x+w]

    face_resized = cv2.resize(face_crop, IMG_SIZE)

    face_normalized = face_resized / 255.0

    return face_normalized.flatten()


# =====================================================
# 2. MEMBACA DATASET
# =====================================================
def load_dataset(dataset_path):

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

                    vector = detect_and_crop_face(image_path)

                    X.append(vector)

                    labels.append(person_name)

                except ValueError as e:

                    print(f"Melewati {filename}: {e}")

    return np.array(X), np.array(labels)


# =====================================================
# 3. TRAINING PCA
# =====================================================
def train_pca_model(dataset_path):

    print("=========================================")
    print("MEMUAT DATASET")
    print("=========================================")

    X_train, labels = load_dataset(dataset_path)

    if len(X_train) == 0:
        raise ValueError(
            f"Tidak ada wajah yang berhasil diproses dari folder:\n{dataset_path}"
        )

    print(f"Jumlah gambar : {len(X_train)}")

    n_components = min(50, len(X_train))

    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X_train)

    print("Training PCA selesai")

    print(
        f"Total Variance : {np.sum(pca.explained_variance_ratio_):.4f}"
    )

    return pca


# =====================================================
# 4. INISIALISASI MODEL
# =====================================================
def initialize_model(dataset_path):
    """
    Fungsi ini dipanggil SATU KALI
    saat aplikasi Streamlit dijalankan.
    """

    global pca_model

    if pca_model is None:

        pca_model = train_pca_model(dataset_path)

    return pca_model


# =====================================================
# 5. MEMBANDINGKAN WAJAH
# =====================================================
def compare_faces(image_path_1,
                  image_path_2,
                  pca_model,
                  threshold=0.75):

    try:

        face_1 = detect_and_crop_face(image_path_1)

        face_2 = detect_and_crop_face(image_path_2)

    except ValueError as e:

        return None, str(e)

    face_1 = face_1.reshape(1, -1)

    face_2 = face_2.reshape(1, -1)

    face_1_pca = pca_model.transform(face_1)

    face_2_pca = pca_model.transform(face_2)

    similarity = cosine_similarity(
        face_1_pca,
        face_2_pca
    )[0][0]

    if similarity >= threshold:

        result = "Mirip"

    else:

        result = "Tidak Mirip"

    return similarity, result


# =====================================================
# 6. FUNGSI KHUSUS UNTUK STREAMLIT
# =====================================================
def compare_uploaded_images(image_path_1,
                            image_path_2,
                            threshold=0.75):
    """
    Streamlit cukup memanggil fungsi ini.
    """

    global pca_model

    if pca_model is None:

        raise ValueError(
            "Model PCA belum diinisialisasi.\n"
            "Jalankan initialize_model(dataset_path) terlebih dahulu."
        )

    return compare_faces(
        image_path_1,
        image_path_2,
        pca_model,
        threshold
    )


# =====================================================
# 7. TEST DARI TERMINAL
# =====================================================
if __name__ == "__main__":

    folder_dataset = r"D:\TEKNIK_INFORMATIKA\SEMESTER 2\ALJABAR LINEAR\UAS\dataset"

    initialize_model(folder_dataset)

    path_wajah_1 = r"D:\TEKNIK_INFORMATIKA\SEMESTER 2\ALJABAR LINEAR\UAS\s3.jpeg"

    path_wajah_2 = r"D:\TEKNIK_INFORMATIKA\SEMESTER 2\ALJABAR LINEAR\UAS\s3.jpeg"

    similarity_score, status = compare_uploaded_images(
        path_wajah_1,
        path_wajah_2,
        threshold=0.75
    )

    print("\n==============================")

    print("HASIL PERBANDINGAN")

    print("==============================")

    if similarity_score is not None:

        print(f"Gambar 1     : {path_wajah_1}")

        print(f"Gambar 2     : {path_wajah_2}")

        print(f"Similarity   : {similarity_score:.4f}")

        print(f"Status       : {status}")

    else:

        print(status)

    print("==============================")
