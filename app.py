
import streamlit as st
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import requests

st.title("تحليل صور Sentinel-2 وعد الدوائر")

st.markdown("""
🎯 هذا التطبيق يتيح لك:
- لصق **رابط مباشر لصورة من Sentinel Hub (EO Browser)**
- تحليل الصورة لاكتشاف وعدّ **الدوائر الزراعية**
- لا حاجة لأي API Key
""")
image_url = st.text_input("📥 أدخل رابط الصورة (JPG/PNG) من EO Browser أو مصدر آخر")

def detect_circles(image_pil):
    image = np.array(image_pil.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        count = len(circles[0, :])
        for i in circles[0, :]:
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
    output_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    return count, output_image

if st.button("🔍 تحليل الصورة"):
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="الصورة الأصلية", use_column_width=True)
            count, result_image = detect_circles(image)
            st.image(result_image, caption=f"الدوائر المكتشفة: {count}", use_column_width=True)
            st.success(f"✅ عدد الدوائر المكتشفة: {count}")
        except Exception as e:
            st.error(f"حدث خطأ في تحميل أو معالجة الصورة: {e}")
    else:
        st.warning("يرجى إدخال رابط صورة صالح.")
