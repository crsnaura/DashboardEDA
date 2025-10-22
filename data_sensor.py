import pandas as pd
import numpy as np

def sensor_nama(nama):
    """
    Mengganti nama dengan karakter bintang (*), mempertahankan huruf pertama dan terakhir.
    Contoh: 'Andi Pratama' menjadi 'A***********a'
    """
    # Menangani nilai NaN (kosong) dan non-string
    if pd.isna(nama) or not isinstance(nama, str) or len(nama.strip()) < 3:
        return "[Nama Disensor]"

    nama_str = nama.strip()
    
    if len(nama_str) <= 2:
        return "[Nama Disensor]"

    # Mengambil huruf pertama dan terakhir
    first_letter = nama_str[0]
    last_letter = nama_str[-1]
    
    # Sensor semua karakter di tengah
    mask_length = len(nama_str) - 2
    mask = '*' * mask_length
    
    return f"{first_letter}{mask}{last_letter}"

def sensor_npm(npm):
    """
    Sensor NPM, mempertahankan 3 digit awal dan 3 digit akhir.
    Contoh: '12111001' menjadi '121##001'
    """
    # Konversi ke string, pastikan tidak ada spasi
    npm_str = str(npm).strip()
    
    # Menangani NPM yang terlalu pendek atau kosong
    if len(npm_str) < 7:
         return "######### [NPM Disensor]"

    # Mengambil 3 digit awal dan 3 digit akhir
    prefix = npm_str[:3]
    suffix = npm_str[-3:]

    # Panjang bagian yang disensor
    mask_length = len(npm_str) - 6
    mask = '#' * mask_length

    return f"{prefix}{mask}{suffix}"
