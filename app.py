import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title("2كشف دوائر الري المحوري فقط من صورة مرفوعة")

st.markdown("""
📸 قم برفع صورة جوية (يفضّل من قمر صناعي) تحتوي على أراضٍ زراعية.
🚜 سيتم اكتشاف وعدّ **دوائر الري المحوري فقط** بناءً على الحجم واللون.
""")

uploaded_file = st.file_uploader("📥 اختر صورة (JPG / PNG)", type=["jpg", "jpeg", "png"])

def is_irrigation_circle(image, circle, min_radius=40, max_radius=150):
    x, y, r = circle
    if not (min_radius < r < max_radius):
        return False

    # اقتطاع منطقة داخل الدائرة لحساب اللون
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    mean_val = cv2.mean(image, mask=mask)[:3]

    # تصنيف اللون كأخضر/بني (نطاق تقريبي)
    green_threshold = (mean_val[1] > 60 and mean_val[1] > mean_val[0] and mean_val[1] > mean_val[2])
    brown_threshold = (mean_val[1] > 40 and mean_val[2] > 40 and mean_val[0] > 40)
    return green_threshold or brown_threshold

def detect_irrigation_circles(image_pil):
    image = np.array(image_pil.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=20, maxRadius=200)

    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        irrigation_circles = [c for c in circles if is_irrigation_circle(image, c)]
        count = len(irrigation_circles)
        for c in irrigation_circles:
            x, y, r = c
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

    output_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    return count, output_image

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 الصورة الأصلية", use_column_width=True)
        count, result_image = detect_irrigation_circles(image)
        st.image(result_image, caption=f"🟢 دوائر الري المكتشفة: {count}", use_column_width=True)
        st.success(f"✅ عدد دوائر الري المحوري: {count}")
    except Exception as e:
        st.error(f"حدث خطأ أثناء المعالجة: {e}")
