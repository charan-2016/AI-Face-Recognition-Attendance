import cv2
import numpy as np
import os

# Load LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load Haar cascade (optional here, since not used for training)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faces = []
ids = []
dataset_path = "dataset"

# Loop through each student's folder
for person_id, person_name in enumerate(os.listdir(dataset_path)):
    person_path = os.path.join(dataset_path, person_name)
    for img_name in os.listdir(person_path):
        if img_name.endswith(".jpg") and not img_name.endswith("_color.jpg"):
            img_path = os.path.join(person_path, img_name)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                faces.append(gray)
                ids.append(person_id)

# Train and save model
recognizer.train(faces, np.array(ids))
recognizer.write("trained_model.yml")

# Save label mapping
with open("labels.txt", "w") as f:
    for name in os.listdir(dataset_path):
        f.write(f"{name}\n")

print("[INFO] Training completed successfully.")
