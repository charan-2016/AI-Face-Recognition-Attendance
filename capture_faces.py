import cv2
import os
import numpy as np

# Step 1: Capture Faces
name = input("Enter Student Name: ")
folder = os.path.join("dataset", name)
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0

print("Capturing faces... Press ESC to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_color = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]

        cv2.imwrite(os.path.join(folder, f"{count}_color.jpg"), face_color)
        cv2.imwrite(os.path.join(folder, f"{count}.jpg"), face_gray)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Capturing Faces', frame)
    if cv2.waitKey(1) == 27 or count >= 10:  # ESC key or 10 faces
        break

cap.release()
cv2.destroyAllWindows()

# Step 2: Train the Model
print("Training model...")

recognizer = cv2.face.LBPHFaceRecognizer_create()
faces, ids = [], []
dataset_path = "dataset"

for person_id, person_name in enumerate(os.listdir(dataset_path)):
    person_path = os.path.join(dataset_path, person_name)
    for img_name in os.listdir(person_path):
        if img_name.endswith(".jpg") and not img_name.endswith("_color.jpg"):
            img_path = os.path.join(person_path, img_name)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                faces.append(gray)
                ids.append(person_id)

recognizer.train(faces, np.array(ids))
recognizer.write("trained_model.yml")

with open("labels.txt", "w") as f:
    for name in os.listdir(dataset_path):
        f.write(f"{name}\n")

print("Training completed successfully!")
