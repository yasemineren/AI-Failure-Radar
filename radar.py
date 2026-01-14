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
**Amaç:** Gelen sensör verilerindeki bozulmaları (Drift) tespit edip, model hata yapmadan önce operatörü uyarmak.
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
st.sidebar.info("Modelin sahadaki şartlarını buradan bozabilirsin.")

# Gürültü ve Kayma Ekleme
noise_amount = st.sidebar.slider("Sinyal Gürültüsü (Noise)", 0.0, 5.0, 0.0, help="Sensörlere binen parazit")
shift_amount = st.sidebar.slider("Veri Kayması (Drift)", 0.0, 5.0, 0.0, help="Düşman kamuflaj değiştirdiğinde veri kayar")

# --- 3. CANLI VERİ AKIŞI ---
# Sahadaki veriyi simüle ediyoruz (Kullanıcının bozduğu veri)
X_current = X_production_base.copy()

# Seçilen bir özelliği bozalım (Örn: 'mean radius' - Hedef boyutu)
target_feature = 'mean radius'
X_current[target_feature] = X_current[target_feature] + np.random.normal(0, noise_amount, len(X_current)) + shift_amount

# --- 4. RADAR ANALİZİ (DRIFT TESPİTİ) ---
st.subheader("📊 Canlı İstihbarat Analizi")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"**Takip Edilen Sinyal:** `{target_feature}`")
    
    # İki veriyi karşılaştır (Eğitim vs Şu An)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Referans (Yeşil - Güvenli)
    plt.hist(X_reference[target_feature], bins=30, alpha=0.5, color='green', label='Referans (Eğitim Verisi)', density=True)
    
    # Canlı (Kırmızı - Şüpheli)
    plt.hist(X_current[target_feature], bins=30, alpha=0.5, color='red', label='Canlı (Saha Verisi)', density=True)
    
    plt.title("Veri Dağılım Analizi (Distribution Drift)")
    plt.legend()
    st.pyplot(fig)

with col2:
    st.markdown("### 🛡️ Durum Raporu")
    
    # 1. İstatistiksel Test (Kolmogorov-Smirnov)
    # Fizikçi gibi düşün: İki dalga fonksiyonu üst üste biniyor mu?
    stat, p_value = ks_2samp(X_reference[target_feature], X_current[target_feature])
    
    # Drift Skoru (0: Aynı, 1: Tamamen Farklı)
    drift_score = stat 
    
    st.metric("Drift Şiddeti", f"{drift_score:.4f}", delta_color="inverse")
    
    # Alarm Mantığı
    threshold = 0.15 # Eşik değer
    
    if drift_score > threshold:
        st.error("🚨 KRİTİK ALARM")
        st.markdown("**Tespit:** Veri karakteristiği bozuldu. Model güvenilmez!")
        status = "FAIL"
    else:
        st.success("✅ SİSTEM STABİL")
        st.markdown("**Tespit:** Veri akışı normal.")
        status = "OK"

# --- 5. MODEL PERFORMANS ETKİSİ ---
st.markdown("---")
st.subheader("🎯 Model İsabet Oranı Etkisi")

# Model şu anki bozuk veriyle ne kadar başarılı?
current_pred = model.predict(X_current)
current_acc = accuracy_score(y_production, current_pred)

col3, col4 = st.columns(2)

with col3:
    st.metric("Modelin Normal Başarısı", f"%{base_acc*100:.2f}")
    
with col4:
    # Başarı düştü mü?
    diff = current_acc - base_acc
    st.metric("Şu Anki Başarı", f"%{current_acc*100:.2f}", delta=f"{diff*100:.2f}%")

if status == "FAIL" and (base_acc - current_acc) > 0.1:
    st.warning("⚠️ DİKKAT: Veri kayması nedeniyle modelin isabet oranı ciddi şekilde düştü. Manuel kontrole geçilmeli.")
