import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd

# ============================================================
# KONFIGURASI
# ============================================================
IMG_SIZE = (150, 150)  # sesuai input_layer model: (None, 150, 150, 3)

# Urutan kelas mengikuti urutan alfabetis folder dataset saat training
# (default ImageDataGenerator / image_dataset_from_directory)
CLASS_NAMES = ["Blackhead", "Cyst", "Papule", "Pustule", "Whitehead"]

# Ciri-ciri tiap kelas jerawat (sumber: Halodoc)
CLASS_INFO = {
    "Blackhead": {
        "ciri": [
            "Warna Gelap: Tampak seperti bintik hitam atau kekuningan di permukaan kulit.",
            "Permukaan Terbuka: Dikenal sebagai komedo terbuka, tidak tertutup oleh lapisan kulit.",
            "Tidak Meradang: Umumnya tidak terasa sakit, nyeri, atau bengkak kecuali terjadi infeksi bakteri.",
            "Tekstur Kulit: Area kulit yang terkena terasa kasar saat diraba.",
        ],
        "sumber": "Halodoc",
    },
    "Whitehead": {
        "ciri": [
            "Bentuk & Warna: Benjolan kecil (1-2 milimeter) berwarna putih atau krem karena berada di bawah lapisan kulit dan tidak terkena udara.",
            "Tekstur: Permukaan kulit terasa kasar atau bergelombang (seperti berpasir) saat diraba.",
            "Tidak Meradang: Berbeda dengan jerawat biasa, area sekitar benjolan tidak membengkak, memerah, atau terasa nyeri.",
        ],
        "sumber": "Halodoc",
    },
    "Pustule": {
        "ciri": [
            "Tampilan Fisik: Benjolan kulit yang membengkak berwarna merah di sekelilingnya, dengan puncak atau titik di bagian tengah yang berwarna putih kekuningan.",
            "Tekstur: Terasa lunak atau sensitif jika disentuh, dengan sedikit sensasi keras di dasarnya.",
            "Rasa Nyeri: Seringkali menimbulkan rasa nyeri atau nyut-nyutan, serta terasa hangat jika disentuh akibat peradangan.",
        ],
        "sumber": "Halodoc",
    },
    "Papule": {
        "ciri": [
            "Padat dan Keras: Terasa solid dan keras saat disentuh.",
            "Tidak Bernanah: Berbeda dengan pustula yang memiliki titik putih berisi nanah, papula murni berupa benjolan keras yang meradang.",
            "Ukuran Kecil: Umumnya berdiameter kurang dari 1 sentimeter atau sekitar beberapa milimeter.",
            "Warna Kemerahan: Area benjolan biasanya berwarna merah (eritema) dan terkadang terasa hangat di kulit.",
        ],
        "sumber": "Halodoc",
    },
    "Cyst": {
        "ciri": [
            "Ukuran Besar: Berupa benjolan yang jauh lebih besar dari jerawat biasa.",
            "Tekstur & Isi: Terasa keras di bagian dalam dan seringkali berisi nanah.",
            "Nyeri: Menimbulkan rasa sakit, nyeri, atau sangat lunak/sensitif saat disentuh.",
            "Warna & Area: Kulit di sekitarnya mengalami kemerahan yang luas.",
        ],
        "sumber": "Halodoc",
    },
}

# Selalu hitung path relatif terhadap lokasi file app.py ini,
# supaya tidak tergantung dari folder mana `streamlit run` dijalankan.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_INFO = {
    "60:20:20 (Train:Val:Test)": os.path.join(BASE_DIR, "models", "60_20_20.keras"),
    "70:15:15 (Train:Val:Test)": os.path.join(BASE_DIR, "models", "70_15_15.keras"),
    "80:10:10 (Train:Val:Test)": os.path.join(BASE_DIR, "models", "80_10_10.keras"),
}

st.set_page_config(
    page_title="Klasifikasi Jenis Jerawat",
    layout="wide",
)

# ============================================================
# LOAD MODEL (cache supaya tidak load ulang setiap interaksi)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    if not os.path.exists(model_path):
        st.error(
            f"File model tidak ditemukan di:\n\n`{model_path}`\n\n"
            f"Pastikan struktur foldernya seperti ini (relatif terhadap app.py):\n"
            f"```\n{os.path.basename(os.path.dirname(model_path))}/\n"
            f"    {os.path.basename(model_path)}\n```\n"
            f"Lokasi app.py saat ini: `{BASE_DIR}`"
        )
        st.stop()
    return tf.keras.models.load_model(model_path, compile=False)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocessing HARUS sama dengan saat training:
    - Resize ke 150x150
    - Konversi ke RGB (3 channel)
    - Tetap dalam skala piksel 0-255 (JANGAN dibagi 255 di sini),
      karena model sudah punya layer Rescaling(1/255) + Normalization
      bawaan EfficientNetB0 di dalamnya.
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image).astype("float32")  # tetap 0-255
    arr = np.expand_dims(arr, axis=0)  # tambah dimensi batch -> (1, 150, 150, 3)
    return arr


def predict(model, image: Image.Image):
    arr = preprocess_image(image)
    preds = model.predict(arr, verbose=0)[0]  # shape (5,)
    return preds


def show_prediction_result(preds: np.ndarray):
    df = pd.DataFrame({
        "Kelas": CLASS_NAMES,
        "Probabilitas (%)": (preds * 100).round(2),
    }).sort_values("Probabilitas (%)", ascending=False)

    top_idx = int(np.argmax(preds))
    top_class = CLASS_NAMES[top_idx]

    st.success(f"Prediksi: {top_class} ({preds[top_idx]*100:.2f}% keyakinan)")

    info = CLASS_INFO.get(top_class)
    if info:
        st.markdown(f"#### Ciri-Ciri {top_class}")
        for ciri in info["ciri"]:
            st.markdown(f"- {ciri}")
        st.caption(f"Sumber: {info['sumber']}")

    st.bar_chart(df.set_index("Kelas"))
    st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Pengaturan")
selected_label = st.sidebar.selectbox("Pilih Model (rasio split data)", list(MODEL_INFO.keys()))
selected_path = MODEL_INFO[selected_label]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Tentang Model**\n\n"
    "- Backbone: EfficientNetB0 (transfer learning)\n"
    "- Input: 150x150 piksel, RGB\n"
    "- Output: 5 kelas jerawat\n"
    "- Preprocessing (resize, rescale, normalisasi) dilakukan otomatis "
    "di dalam model — gambar input cukup RGB 0-255."
)

# ============================================================
# MAIN
# ============================================================
st.title("Klasifikasi Jenis Jerawat")
st.write(
    "Unggah foto area kulit berjerawat, lalu pilih model di sidebar untuk "
    "melihat hasil klasifikasi."
)

uploaded_file = st.file_uploader(
    "Unggah gambar (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)
    with col_info:
        st.write(f"**Ukuran asli:** {image.size[0]} x {image.size[1]} px")
        st.write(f"**Akan di-resize ke:** {IMG_SIZE[0]} x {IMG_SIZE[1]} px")
        st.write(f"**Mode warna:** RGB")

    run = st.button("Jalankan Klasifikasi", type="primary")

    if run:
        with st.spinner(f"Memuat model dan memprediksi ({selected_label})..."):
            model = load_model(selected_path)
            preds = predict(model, image)
        st.subheader(f"Hasil — Model {selected_label}")
        show_prediction_result(preds)
else:
    st.info("Silakan unggah gambar terlebih dahulu untuk memulai klasifikasi.")

st.markdown("---")