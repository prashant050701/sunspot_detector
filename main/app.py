import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import streamlit as st
import plotly.express as px

def detect_sunspots(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image_np = np.array(image)

    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, sun_mask = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    sun_only = cv2.bitwise_and(image_np, image_np, mask=sun_mask)

    top_crop = 130
    bottom_crop = 130
    sun_cropped = sun_only[top_crop:-bottom_crop, :]

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 100
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(cv2.cvtColor(sun_cropped, cv2.COLOR_RGB2GRAY))

    data = {
        "x": [],
        "y": [],
        "radius": [],
        "number": [],
    }
    for i, keypoint in enumerate(keypoints):
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        radius = int(keypoint.size / 2) + 5
        data["x"].append(x)
        data["y"].append(y)
        data["radius"].append(radius)
        data["number"].append(i + 1)

    return sun_cropped, data

st.title("Almost live sunspot detection")
st.write("You can use the dropdown below to select and zoom into a specific sunspot.")

sun_image_url = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_4096_HMIIF.jpg"

sun_cropped, sunspot_data = detect_sunspots(sun_image_url)

fig = px.imshow(sun_cropped)
for x, y, radius, number in zip(sunspot_data["x"], sunspot_data["y"], sunspot_data["radius"], sunspot_data["number"]):
    fig.add_shape(type="circle", x0=x-radius, y0=y-radius, x1=x+radius, y1=y+radius,
                  line=dict(color="red", width=2))
    fig.add_annotation(x=x+15, y=y+15, text=str(number), showarrow=False, font=dict(color="blue", size=12))

fig.update_layout(
    dragmode="zoom",
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis={"visible": False},
    yaxis={"visible": False},
)

st.plotly_chart(fig, use_container_width=True)

sunspot_numbers = sunspot_data["number"]
selected_sunspot = st.selectbox("Select a Sunspot Number to Zoom", sunspot_numbers)

selected_index = sunspot_numbers.index(selected_sunspot)
selected_x = sunspot_data["x"][selected_index]
selected_y = sunspot_data["y"][selected_index]
selected_radius = sunspot_data["radius"][selected_index] + 20

zoomed_image = sun_cropped[max(0, selected_y - selected_radius):min(sun_cropped.shape[0], selected_y + selected_radius),
                           max(0, selected_x - selected_radius):min(sun_cropped.shape[1], selected_x + selected_radius)]

st.image(zoomed_image, caption=f"Zoomed View of Sunspot #{selected_sunspot}", use_container_width=True)
