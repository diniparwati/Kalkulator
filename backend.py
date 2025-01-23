from flask import Flask, request, render_template, url_for, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import logging

# Inisialisasi Flask
app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

# Path ke model
MODEL_PATH = "D:\ProjectDL\burung.h5"

# Muat model .h5
try:
    model = keras.models.load_model(MODEL_PATH)
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    model = None

# Kelas target
class_names = ['Nektavior', 'Predator', 'Seedivora']

# Fungsi untuk melakukan prediksi
def predict_image(image_path):
    try:
        img_height, img_width = 150, 150  # Sesuaikan dengan ukuran input model
        # Muat gambar dan ubah ukuran
        img = keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
        img_array = keras.preprocessing.image.img_to_array(img)  # Ubah ke array
        img_array = tf.expand_dims(img_array, 0)  # Tambahkan batch dimension

        # Lakukan prediksi
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])  # Hitung softmax
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        return predicted_class, confidence
    except Exception as e:
        print(f"Error saat melakukan prediksi: {e}")
        raise e

# Rute untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Rute untuk melakukan klasifikasi
@app.route('/prediksi', methods=['POST'])
def classify():
    try:
        # Validasi apakah ada file yang diunggah
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Validasi tipe file (hanya gambar yang diperbolehkan)
        if not file.content_type.startswith("image/"):
            return jsonify({"error": "Invalid file type. Please upload an image."}), 400

        # Simpan file ke folder static/uploads
        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Lakukan prediksi
        predicted_class, confidence = predict_image(file_path)

        # Kirim hasil prediksi dalam format JSON
        return jsonify({
            "result": predicted_class,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        # Tangkap error dan kirim respons error ke frontend
        print(f"Error saat memproses permintaan: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Main
if __name__ == '__main__':
    app.run(debug=True)