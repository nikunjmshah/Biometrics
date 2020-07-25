# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
from statistics import mean

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-id", "--name", required=True,
	help="ID of input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

input_encodings = args["encodings"] + "/enc_" + name
input_image = args["image"]

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(input_encodings, "rb").read())

# load the input image and convert it from BGR to RGB
image = cv2.imread(input_image)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb, model=args["detection-method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# loop over the facial embeddings
for encoding in encodings:
    face_distances = face_recognition.face_distance(data["encodings"], encoding)
    print("Mean distance = " + str(mean(face_distances)))
    print("Face distance array:")
    print(face_distances)
    if mean(face_distances) < 0.6:
        print("Face-ID matched")
    else:
        print("Face-ID not matched")