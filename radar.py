import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="AI Defense Radar", layout="wide", page_icon="📡")

st.title("📡 AI Model Güvenlik & Gözlem Radarı")
st.markdown("""
**Senaryo:** Sahadaki bir İHA'nın Dost/Düşman tanıma sistemi.
**Amaç:** Sensör verilerindeki **Topyekün Karıştırmayı (Global Jamming)** ve veri kaymasını (Drift) tespit edip, model hata yapmadan önce operatörü uyarmak.
""")

# --- 1. MODEL EĞİTİMİ (SİMÜLASYON) ---
@st.cache_resource
def build_defense_model():
    # Veriyi yükle (Meme Kanseri verisi -> Savunma için 'Tehdit Tespiti' olarak düşünelim)
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Eğitim ve Test olarak ayır
    # X_ref: Eğitimde gördüğü "Temiz" veri
    # X_prod: Sahaya çıktığında karşılaşacağı veri
    X_ref, X_prod, y_ref, y_prod = train_test_split(df, y, test_size=0.5, random_state=42)
    
    # Modeli eğit
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_ref, y_ref)
    
    # Referans başarısını ölç
    base_accuracy = accuracy_score(y_ref, model.predict(X_ref))
    
    return model, X_ref, X_prod, y_prod, base_accuracy

model, X_reference, X_production_base, y_production, base_acc = build_defense_model()

# --- 2. SABOTAJ PANELİ (YAN MENÜ) ---
st.sidebar.header("⚔️ Elektronik Harp (Sabotaj)")
st.sidebar.info("Modelin tüm sensörlerine buradan gürültü basabilirsin.")

# Gürültü ve Kayma Ekleme
noise_amount = st.sidebar.slider("Sinyal Gürültüsü (Noise)", 0.0, 5.0, 0.0, help="Tüm sensörlere binen parazit şiddeti")
shift_amount = st.sidebar.slider("Veri Kayması (Drift)", 0.0, 3.0, 0.0, help="Veri dağılımını kaydırma katsayısı")

# --- 3. CANLI VERİ AKIŞI (GLOBAL JAMMING) ---
# Sahadaki veriyi simüle ediyoruz
X_current = X_production_base.copy()

# ARTIK TEK BİR SÜTUNU DEĞİL, TÜM VERİYİ BOZUYORUZ
# Her özelliğin (sütunun) kendi yapısına göre gürültü ekliyoruz
for col in X_current.columns:
    # O sütunun standart sapmasını alıp, gürültüyü ona göre ölçekliyoruz
    # Böylece küçük sayılar (0.01) ile büyük sayılar (1000) orantılı bozulur
    std_dev = X_current[col].std()
    mean_val = X_current[col].mean()
    
    # Gürültü üret (Noise)
    noise = np.random.normal(0, std_dev * noise_amount, len(X_current))
    
    # Kayma üret (Drift)
    shift = mean_val * shift_amount
    
    # Veriyi boz
    X_current[col] = X_current[col] + noise + shift

# Görselleştirme için yine sadece tek bir özelliği seçip gösterelim (Temsili)
target_feature = 'mean radius'

# --- 4. RADAR ANALİZİ (DRIFT TESPİTİ) ---
st.subheader("📊 Canlı İstihbarat Analizi")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"**Spektrum Analizi:** `{target_feature}` (Temsili Kanal)")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    # Referans (Yeşil - Güvenli)
    plt.hist(X_reference[target_feature], bins=30, alpha=0.5, color='green', label='Referans (Eğitim Verisi)', density=True)
    # Canlı (Kırmızı - Şüpheli)
    plt.hist(X_current[target_feature], bins=30, alpha=0.5, color='red', label='Canlı (Bozuk Veri)', density=True)
    
    plt.title(f"Sinyal Dağılımı: {target_feature}")
    plt.legend()
    st.pyplot(fig)

with col2:
    st.markdown("### 🛡️ Durum Raporu")
