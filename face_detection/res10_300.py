import cv2
import numpy as np
from imutils.video import FileVideoStream
import imutils
import time 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False, default='example/example_01.jpg' ,
                help='path to image file')
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	            help="minimum probability to filter weak detections")
args = ap.parse_args()

# To-do:
# Handle both image and video 
prototxt_detection = "model/res-10_300x300/deploy.prototxt"
model_detection = "model/res-10_300x300/res10_300x300_ssd_iter_140000.caffemodel"

# Load res10 prototxt and model
net = cv2.dnn.readNet(prototxt_detection, model_detection)

# Read a image from the disk
image = cv2.imread(args.image)
(h, w) = image.shape[:2]

# Construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# Compute face detection
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > args.confidence:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

# Display output image
cv2.imshow("face detection with res-10", image)
cv2.waitKey()

# Close the winddow
cv2.destroyAllWindows()