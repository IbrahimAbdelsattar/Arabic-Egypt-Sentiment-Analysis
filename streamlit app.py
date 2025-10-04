import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# تحميل الـ tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# تحميل الموديل
model = tf.keras.models.load_model("sentiment_model.h5")

# إعدادات الإدخال
st.title("📊 Arabic Sentiment Analysis")
st.markdown("أدخل نص عربي لتحليل المشاعر (إيجابي / سلبي / محايد).")

user_input = st.text_area("أكتب النص هنا:")

if st.button("تحليل"):
    if user_input.strip() != "":
        # تحويل النص إلى sequences
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=100)  # لازم تخلي maxlen نفس اللي كنت مستخدمه في التدريب

        # التنبؤ
        prediction = model.predict(padded)
        sentiment_class = np.argmax(prediction)

        if sentiment_class == 0:
            st.success("😡 سلبي")
        elif sentiment_class == 1:
            st.success("😐 محايد")
        else:
            st.success("😊 إيجابي")
    else:
        st.warning("من فضلك أدخل نص.")
