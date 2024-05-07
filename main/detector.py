import requests
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO

url = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_4096_HMIIF.jpg"
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

for i, keypoint in enumerate(keypoints):
    x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
    radius = int(keypoint.size / 2) + 5  #enlarging the circle
    cv2.circle(sun_cropped, (x, y), radius, (0, 0, 255), 2)
    #cv2.putText(sun_cropped, str(i + 1), (x + 15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


total_sunspots = len(keypoints)
print(f"Total number of sunspots detected: {total_sunspots}")

plt.figure(figsize=(10, 10))
plt.imshow(sun_cropped)
plt.axis('off')
plt.savefig('sun_cropped_numbered.png')
plt.show()
