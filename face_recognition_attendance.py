import cv2
import numpy as np
import pandas as pd
import datetime
import os

# Load trained recognizer and cascade classifier
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model.yml")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load student names
with open("labels.txt", "r") as f:
    names = [line.strip() for line in f.readlines()]

# Initialize attendance set
attendance = set()
cap = cv2.VideoCapture(0)

print("[INFO] Face recognition started. Press ESC to stop...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 60:
                name = names[id_]
                date = datetime.datetime.now().strftime("%Y-%m-%d")

                if (name, date) not in attendance:
                    attendance.add((name, date))
                    print(f"[INFO] {name} marked present on {date}")

                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        # ESC key to exit
        key = cv2.waitKey(1)
        if key == 27:
            print("[INFO] ESC pressed. Exiting...")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    if attendance:
        # Save attendance
        data = []
        for name, date in attendance:
            day = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%A")
            data.append((name, date, day))

        df = pd.DataFrame(data, columns=["Name", "Date", "Day"])
        file_exists = os.path.exists("attendance.csv")
        df.to_csv("attendance.csv", mode='a', index=False, header=not file_exists)
        print("[INFO] Attendance saved to attendance.csv.")
    else:
        print("[INFO] No attendance recorded.")
