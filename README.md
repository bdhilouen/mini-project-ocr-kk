# Aplikasi Web OCR Kartu Keluarga (KK)

Aplikasi mini project berbasis **Python (Flask + SQLite + Jinja2 + EasyOCR)** untuk membaca gambar Kartu Keluarga (KK), meninjau dan mengoreksi hasil pembacaan OCR, lalu menyimpannya ke database SQLite dengan relasi `1 KK : N Anggota Keluarga`.

## Struktur Project

```text
mini_project_citra/
├── app.py                  # Aplikasi Flask, validasi form, dan seluruh route CRUD
├── requirements.txt        # Daftar dependensi Python
├── kk_database.db          # Database SQLite (otomatis dibuat saat aplikasi berjalan)
├── models/
│   ├── __init__.py
│   ├── database.py         # Skema tabel & operasi CRUD SQLite (ON DELETE CASCADE)
│   └── ocr_kk.py           # Preprocessing OpenCV/Pillow + EasyOCR + parser spasial KK
├── static/
│   ├── css/style.css       # Gaya antarmuka bersih berbasis Bootstrap 5
│   └── js/app.js           # Pratinjau gambar, tambah/hapus baris anggota, konfirmasi hapus
├── templates/
│   ├── base.html
│   ├── index.html          # Halaman input upload gambar KK
│   ├── konfirmasi.html     # Halaman konfirmasi & koreksi hasil OCR (editable)
│   ├── daftar_kk.html      # Halaman daftar identitas KK
│   ├── detail_kk.html      # Halaman detail KK dan tabel anggota keluarga
│   ├── edit_kk.html        # Form edit identitas KK
│   └── anggota_form.html   # Form tambah/edit anggota keluarga
├── tests/
│   └── test_app.py         # Unit test & integration test
└── uploads/                # Penyimpanan sementara gambar KK yang diunggah
```

## Cara Menjalankan

1. **Buat dan aktifkan virtual environment (disarankan Python 3.12 atau 3.13):**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install seluruh dependensi:**

   ```powershell
   pip install -r requirements.txt
   ```

3. **Jalankan server Flask:**

   ```powershell
   python app.py
   ```

4. **Buka di browser:**

   `http://127.0.0.1:5000`

> **Catatan EasyOCR:** Saat pertama kali melakukan upload dan proses OCR, EasyOCR akan mengunduh model bahasa Indonesia (`id`) dan Inggris (`en`) secara otomatis, sehingga proses OCR pertama membutuhkan koneksi internet dan waktu sedikit lebih lama. Proses berikutnya menggunakan cache lokal.

## Menjalankan Pengujian

```powershell
python -m unittest discover -s tests -v
```
