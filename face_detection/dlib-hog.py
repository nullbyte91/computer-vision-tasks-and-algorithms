import cv2
import dlib
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False, default='example/example_01.jpg' ,
                help='path to image file')
ap.add_argument('-w', '--weights', required=False, default='./model/mmod_human_face_detector.dat',
                help='path to weights file')
args = ap.parse_args()

# Load input image
image = cv2.imread(args.image)

if image is None:
    print("Could not read input image")
    exit()
    
# Initialize hog + svm based face detector
hog_face_detector = dlib.get_frontal_face_detector()

# CNN Based Face detector
detection = hog_face_detector(image, 1)

# Loop over detected faces
for face in detection:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    # draw box over face
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

# Display output image
cv2.imshow("face detection with dlib", image)
cv2.waitKey()

# Close the winddow
cv2.destroyAllWindows()