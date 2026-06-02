import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Memuat dataset mentah dari jalur file yang ditentukan."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {file_path}")
    df = pd.read_csv(file_path, encoding='latin-1', delimiter=';')
    print("✓ Dataset berhasil dimuat.")
    return df
    

def preprocess_data(df):
    """Melakukan pembersihan data, penanganan outlier, encoding, dan standarisasi."""
    df_clean = df.copy()
    
    # 1. Menghapus duplikat dan memastikan kolom krusial ada (otomatis buat jika hilang)
    df_clean = df_clean.drop_duplicates()
    required_cols = ['Age']
    missing_required = [c for c in required_cols if c not in df_clean.columns]
    if missing_required:
        num_df = df_clean.select_dtypes(include=[np.number])
        if not num_df.empty:
            global_median = num_df.median().median()
        else:
            global_median = 0
        for col in missing_required:
            df_clean[col] = global_median
        print(f"⚠️ Kolom tidak ditemukan dan diisi otomatis: {missing_required} (nilai: {global_median})")
    # Buang baris yang masih memiliki NA pada kolom yang dibutuhkan
    df_clean = df_clean.dropna(subset=required_cols)
    
    # 2. Penanganan Outlier Berbasis DataFrame (Vectorized IQR)
    columns_to_filter = ['Age']
    for col in columns_to_filter:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_limit = Q1 - (1.5 * IQR)
            upper_limit = Q3 + (1.5 * IQR)
            
            # Filter langsung secara efisien
            df_clean = df_clean[(df_clean[col] >= lower_limit) & (df_clean[col] <= upper_limit)]
            
    # 3. Transformasi Data Kategorikal (One-Hot Encoding)
    if 'Preferred_Foot' in df_clean.columns:
        df_clean = pd.get_dummies(df_clean, columns=['Preferred_Foot'], drop_first=True)
        
    # 4. Standarisasi Fitur Numerik
    scaler = StandardScaler()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    # Kecualikan kolom hasil encoding (0 atau 1) dari standarisasi
    cols_to_scale = [c for c in numeric_cols if not c.startswith('Preferred_Foot_')]
    
    if cols_to_scale:
        df_clean[cols_to_scale] = scaler.fit_transform(df_clean[cols_to_scale])
        
    print("✓ Proses preprocessing selesai.")
    return df_clean
# ...existing code...

def main():
    # Menentukan jalur file secara dinamis relatif terhadap struktur folder proyek
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'football_raw', 'football_player_stats.csv')
    output_dir = os.path.join(base_dir, 'preprocessing')
    output_path = os.path.join(output_dir, 'football_preprocessing.csv')
    
    # Menjalankan Pipeline Preprocessing
    try:
        raw_data = load_data(input_path)
        cleaned_data = preprocess_data(raw_data)
        
        # Memastikan folder output tersedia
        os.makedirs(output_dir, exist_ok=True)
        
        # Menyimpan hasil data bersih
        cleaned_data.to_csv(output_path, index=False)
        print(f"✓ Dataset hasil transformasi berhasil disimpan di: {output_path}")
        print(f"Dimensi data akhir: {cleaned_data.shape}")
        
    except Exception as e:
        print(f"❌ Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()