import streamlit as st
import pickle

# -----------------------------
# تحميل الموديل والـ TF-IDF Vectorizer
# -----------------------------
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer (1).pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -----------------------------
# إعداد واجهة Streamlit
# -----------------------------
st.set_page_config(
    page_title="تحليل المشاعر - Sentiment Analysis",
    page_icon="📊",
    layout="centered"
)

st.title("📊 تطبيق تحليل المشاعر بالعربي")
st.markdown(
    """
    👋 أهلاً بيك!  
    التطبيق ده بيحلل النصوص العربية ويقول إذا كان **إيجابي 😊** أو **سلبي 😡** أو **محايد 😐**.  

    جرب تكتب جملة زي:  
    - "التجربة دي حلوة جداً"  
    - "المنتج وحش ومش عجبني"  
    - "الموضوع عادي خالص"  
    """
)

# -----------------------------
# إدخال المستخدم
# -----------------------------
user_input = st.text_area("📝 اكتب الجملة بتاعتك هنا:", "الفيلم كان رائع جداً وممتع 👌")

if st.button("✨ حلّل الجملة"):
    if user_input.strip() != "":
        # تحويل النصوص لتمثيل عددي باستخدام TF-IDF
        new_vec = vectorizer.transform([user_input])

        # التنبؤ
        prediction = model.predict(new_vec)[0]

        # عرض النتيجة
        if prediction == 0:
            st.error("التحليل: 😡 سلبي")
        elif prediction == 1:
            st.warning("التحليل: 😐 محايد")
        else:
            st.success("التحليل: 😊 إيجابي")
    else:
        st.warning("⚠️ من فضلك اكتب جملة عشان نقدر نحللها.")
