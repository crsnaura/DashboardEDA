import pandas as pd
import numpy as np

# Fungsi untuk sensor kolom 'Nama'
def sensor_nama(nama):
    """
    Mengganti nama dengan karakter bintang (*), mempertahankan huruf pertama dan terakhir.
    Contoh: 'Andi' menjadi 'A**i'
    """
    # Menangani nilai NaN (kosong)
    if pd.isna(nama) or not isinstance(nama, str) or len(nama.strip()) < 3:
        return "[Nama Disensor]"

    nama_str = nama.strip()
    
    # Jika panjang kurang dari 3, ganti saja dengan sensor
    if len(nama_str) <= 2:
        return "[Nama Disensor]"

    # Mengambil huruf pertama dan terakhir
    first_letter = nama_str[0]
    last_letter = nama_str[-1]
    
    # Panjang bagian yang disensor (total panjang dikurangi 2 huruf)
    mask_length = len(nama_str) - 2
    mask = '*' * mask_length
    
    return f"{first_letter}{mask}{last_letter}"

# Fungsi untuk sensor kolom 'NPM' (Nomor Pokok Mahasiswa)
def sensor_npm(npm):
    """
    Sensor NPM, mempertahankan 3 digit awal dan 3 digit akhir.
    Contoh: '12111001' menjadi '121##001'
    """
    # Konversi ke string dan hapus spasi jika ada
    npm_str = str(npm).strip()
    
    # Menangani NPM yang terlalu pendek
    if len(npm_str) < 7:
         return "######### [NPM Disensor]"

    # Mengambil 3 digit awal dan 3 digit akhir
    prefix = npm_str[:3]
    suffix = npm_str[-3:]

    # Panjang bagian yang disensor
    mask_length = len(npm_str) - 6
    mask = '#' * mask_length

    return f"{prefix}{mask}{suffix}"


# Blok ini hanya akan dijalankan jika Anda menjalankan file ini secara langsung (python data_sensor.py)
# Jika file ini di-import oleh file lain (app.py), blok ini TIDAK akan dijalankan.
if __name__ == "__main__":
    # --- Data Dummy (Contoh data yang akan disensor) ---
    data = {
        'Nama': ['Andi Pratama', 'Budi Santoso', 'Citra Dewi', 'Doni Kurniawan', 'Eka Farida', 'Gus'],
        'NPM': ['12111001', '12111002', '1211200345', '12113004', '12114005', '987'],
        'Nilai': [85, 92, 78, 88, 95, 80]
    }
    df = pd.DataFrame(data)

    print("--- DataFrame Asli ---")
    print(df.to_markdown(index=False))
    print("\n" + "="*50 + "\n")

    # --- Melakukan Sensor ---
    # Membuat salinan DataFrame agar data asli tetap utuh
    df_sensored = df.copy()

    # Menerapkan fungsi sensor pada kolom 'Nama' dan 'NPM'
    df_sensored['Nama'] = df_sensored['Nama'].apply(sensor_nama)
    df_sensored['NPM'] = df_sensored['NPM'].apply(sensor_npm)

    # --- Tampilkan Hasil Sensor ---
    print("--- DataFrame Setelah Sensor ---")
    print(df_sensored.to_markdown(index=False))
