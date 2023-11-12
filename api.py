import streamlit as st
import joblib
import pandas as pd
import numpy as np


# https://www.webfx.com/tools/emoji-cheat-sheet/ den emoji koydum
st.set_page_config(page_title="Diabetes Detector", page_icon=":candy:", layout="wide")

st.subheader("Diyabet Bulucu")
st.title("Diyabet misiniz hemmen öğrenin")

model = joblib.load('knn.pkl')

Pregnancies = st.sidebar.number_input("Hamilelik Sayısı", help="Toplamda kaç kez hamile kalındı", min_value=0, max_value=20, value=0, step=1)
Glucose = st.sidebar.number_input("Glikoz değeri", help="0-200 arası bir değer giriniz.", min_value=0, max_value=200, value=0, step=1)
BloodPressure = st.sidebar.number_input("Kan Basınç Değeri", help="0-150 arası bir değer giriniz.", min_value=0, max_value=150, value=0, step=1)
SkinThickness = st.sidebar.number_input("Deri Kalınlığı", help="0-99 mm arası bir değer giriniz", min_value=0, max_value=99, value=0, step=1)
Insulin = st.sidebar.number_input("İnsülin", help="0-1000 arası bir değer giriniz",min_value=0, max_value=1000, value=0, step=1)
BMI = st.sidebar.number_input("Vücut Kitle Endeksi", help="0-100 arası bir değer giriniz", min_value=0, max_value=100, value=0, step=1)
DiabetesPedigreeFunction = st.sidebar.number_input("Soyağacındaki Diyabet Değeri", help="0.00 ila 2.50 arası bir değer giriniz")
Age = st.sidebar.number_input("Yaş", help="0-100 arası bir değer giriniz.", min_value=0, max_value=100, value=0, step=1)


input_df = pd.DataFrame({
    'Pregnancies': [Pregnancies],
    'Glucose': [Glucose],
    'BloodPressure': [BloodPressure],
    'SkinThickness': [SkinThickness],
    'Insulin': [Insulin],
    'BMI': [BMI],
    'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
    'Age': [Age]
})

pred = model.predict(input_df.values)
pred_probability = np.round(model.predict_proba(input_df.values), 2)

st.header("Sonuçlar")

# Sonuç Ekranı
if st.sidebar.button("Submit"):

    # Info mesajı oluşturma
    st.info("Sonuçlar aşağıdaki gibidir:")

    results_df = pd.DataFrame({
        'Prediction': [pred],
        'Diabet': [pred_probability[:, :1]],
        'Normal': [pred_probability[:, 1:]]
    })

    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("0","Normal"))
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("1","Diyabet"))



    if pred == 0:
        st.info("Kişi normaldir")
    else:
        st.info("Kişi diyabet hastasıdır.")