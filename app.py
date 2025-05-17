
import streamlit as st
import requests
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

st.title("عد الدوائر في الصور الفضائية")
st.markdown("🚀 أدخل إحداثيات المنطقة وسيتم تحميل صورة أقمار صناعية وتحليلها.")

lat = st.number_input("خط العرض (Latitude)", value=27.0377)
lon = st.number_input("خط الطول (Longitude)", value=14.4283)
zoom = st.slider("مستوى التقريب (Zoom)", min_value=10, max_value=20, value=15)
api_key = st.text_input("Google Maps API Key", type="password")
size = "640x640"

def get_satellite_image(lat, lon, zoom, size, api_key):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={size}&maptype=satellite&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return None

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

if st.button("تحميل الصورة وتحليلها"):
    if api_key:
        image_pil = get_satellite_image(lat, lon, zoom, size, api_key)
        if image_pil:
            st.image(image_pil, caption="الصورة الأصلية", use_column_width=True)
            count, result_image = detect_circles(image_pil)
            st.image(result_image, caption=f"الدوائر المكتشفة: {count}", use_column_width=True)
            st.success(f"✅ عدد الدوائر المكتشفة: {count}")
        else:
            st.error("فشل تحميل الصورة. تحقق من المفتاح أو الاتصال بالإنترنت.")
    else:
        st.warning("يرجى إدخال مفتاح API أولاً.")
