# =====================================
# TUNADROMD Veri Seti - Model Eğitimi
# =====================================

import pandas as pd
import numpy as np
import os
import joblib
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from urllib.request import urlopen

# ------------------------------------------------
# 1) VERİYİ İNDİR
# ------------------------------------------------

print("\n⏳ Veri indiriliyor...")

url = "https://archive.ics.uci.edu/static/public/813/tunadromd.zip"

zip_path = "tunadromd.zip"

# URL'den zip dosyasını indir
import urllib.request
urllib.request.urlretrieve(url, zip_path)

print("✔ Veri indirildi.")

# ZIP'i aç
import zipfile
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("tunadromd_data")

print("✔ ZIP açıldı.")

# -------------------------------------------------
# 2) VERİYİ OKU
# -------------------------------------------------
print("\n📌 Veri okunuyor...")

# Veri dosyasının adı:
csv_files = glob.glob("tunadromd_data/*.csv")

if len(csv_files) == 0:
    raise FileNotFoundError("CSV dosyası bulunamadı. 'tunadromd_data' klasörünü kontrol et.")

data_path = csv_files[0]
print("📄 Bulunan CSV:", data_path)


df = pd.read_csv(data_path)

print("✔ Veri başarıyla yüklendi.")
print("\nVeri Boyutu:", df.shape)
print(df.head())

# -----------------------------------------------
# 3) HEDEF VE ÖZELLİKLERİ AYIR
# -----------------------------------------------

target_col = "Label"  # dataset'teki hedef

df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

print("\n🎯 Hedef sınıflar:", y.unique())

# -------------------------------------------------
# 4) TRAIN/TEST BÖLME
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n✔ Train/Test bölündü.")

# -------------------------------------------------
# 5) PIPELINE OLUŞTUR: SCALE → FEATURE SELECT → MODEL
# -------------------------------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif, k=10)),  # her feature’ın hedef değişkenle ilişkisine göre sıralama yapılıp, en yüksek skora sahip ilk 10 özellik sabit olarak seçilir.
    ("model", RandomForestClassifier(random_state=42)) 
])

# -------------------------------------------------
# 6) HYPERPARAMETER TUNING (GridSearchCV)
# -------------------------------------------------

print("\n🔍 Hyperparameter Search başlıyor...")

param_grid = {                           # en iyi performansın hangi bölgeye yakın olduğunu bulmak için
    "model__n_estimators": [100, 300],   # 2 değer
    "model__max_depth": [None, 10, 20],  # 3 değer
    "model__min_samples_split": [2, 5]   # 2 değer
}
#Kombinasyon sayısı = (değer1 sayısı) × (değer2 sayısı) × (değer3 sayısı) == 2*3*2 = 12 farklı model eder

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,                    # veri 3 parçaya bölünür , her kombinasyon 3 kez farklı train/test ile denenir
    scoring="accuracy",      # modelleri doğruluk skoruna göre değerlendirmek için
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\n✔ En iyi skor:", grid.best_score_)
print("✔ En iyi parametreler:", grid.best_params_)

best_model = grid.best_estimator_

# -----------------------------------------------
# 7) TEST SETİNDE DEĞERLENDİRME
# -----------------------------------------------

print("\n📊 Test Set Performansı:")

y_pred = best_model.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------------------------
# 8) MODELİ KAYDET
# -----------------------------------------------

save_path = "TUNADROMD_RF_Model.pkl"
joblib.dump(best_model, save_path)

print(f"\n💾 Model kaydedildi: {save_path}")

print("\n🎉 İşlem tamamlandı. Model başarıyla eğitildi!")

