import requests
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO

url = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_4096_HMII.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert('L')

image_np = np.array(image)
kernel = np.array([[-1, -1, -1], [-1, 8.25, -1], [-1, -1, -1]])
convoluted = cv2.filter2D(image_np, -1, kernel)

_, binary_threshold = cv2.threshold(convoluted, 50, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

image_with_boxes = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 5 and h > 5:
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)

plt.figure(figsize=(10, 10))
plt.imshow(image_with_boxes)
plt.axis('off')
plt.savefig('sun.png')
plt.show()
