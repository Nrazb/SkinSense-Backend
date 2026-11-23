
import cv2
import matplotlib.pyplot as plt

from app.ai.computer_vision.preprocessing import preprocess_image


# Lokasi gambar pengujian
IMAGE_PATH = "dataset/classification/test_face.jpeg"


# Membaca gambar
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Gambar tidak ditemukan: {IMAGE_PATH}"
    )


# Menjalankan preprocessing
result = preprocess_image(image)


# Menampilkan hasil pengujian di terminal
print("Preprocessing berhasil!")
print("Ukuran gambar:", result.shape)
print("Tipe data:", result.dtype)


# Mengubah gambar asli dari BGR menjadi RGB
original_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# Menampilkan gambar asli dan hasil preprocessing
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Gambar Asli")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title("Hasil Preprocessing")
plt.axis("off")

plt.tight_layout()
plt.show()
