import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Almost live sunspot detection", layout="wide")

@st.cache_data(show_spinner=False)
def fetch_image(url):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    return np.array(img)

def detect_sunspots(url):
    image_np = fetch_image(url)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, sun_mask = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    sun_only = cv2.bitwise_and(image_np, image_np, mask=sun_mask)
    top_crop = 130
    bottom_crop = 130
    sun_cropped = sun_only[top_crop:-bottom_crop, :]
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 100
    params.filterByArea = True
    params.minArea = 20
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(cv2.cvtColor(sun_cropped, cv2.COLOR_RGB2GRAY))
    data = {"x": [], "y": [], "radius": [], "number": []}
    for i, k in enumerate(keypoints):
        x, y = int(k.pt[0]), int(k.pt[1])
        radius = int(k.size / 2) + 5
        data["x"].append(x)
        data["y"].append(y)
        data["radius"].append(radius)
        data["number"].append(i + 1)
    return sun_cropped, data

st.title("Almost live sunspot detection")
st.write("You can use the dropdown below to select and zoom into a specific sunspot.")

sun_image_url = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_4096_HMIIF.jpg"

try:
    sun_cropped, sunspot_data = detect_sunspots(sun_image_url)
except Exception as e:
    st.error(f"Failed to fetch/process image: {e}")
    st.stop()

fig = px.imshow(sun_cropped)
for x, y, radius, number in zip(sunspot_data["x"], sunspot_data["y"], sunspot_data["radius"], sunspot_data["number"]):
    fig.add_shape(type="circle", x0=x-radius, y0=y-radius, x1=x+radius, y1=y+radius, line=dict(width=2))
    fig.add_annotation(x=x+15, y=y+15, text=str(number), showarrow=False)

fig.update_layout(dragmode="zoom", margin=dict(l=0, r=0, t=0, b=0))
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

st.plotly_chart(fig, use_container_width=True)

if len(sunspot_data["number"]) == 0:
    st.info("No sunspots detected with current settings.")
else:
    selected_sunspot = st.selectbox("Select a Sunspot Number to Zoom", sunspot_data["number"])
    idx = sunspot_data["number"].index(selected_sunspot)
    sx = sunspot_data["x"][idx]
    sy = sunspot_data["y"][idx]
    sr = sunspot_data["radius"][idx] + 20
    y0 = max(0, sy - sr)
    y1 = min(sun_cropped.shape[0], sy + sr)
    x0 = max(0, sx - sr)
    x1 = min(sun_cropped.shape[1], sx + sr)
    zoomed_image = sun_cropped[y0:y1, x0:x1]
    st.image(zoomed_image, caption=f"Zoomed View of Sunspot #{selected_sunspot}")
