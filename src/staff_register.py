import cv2
import sqlite3
import numpy as np
from insightface.app import FaceAnalysis

# Load InsightFace model
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

# Connect database
conn = sqlite3.connect("database/customers.db")
cursor = conn.cursor()

# Input staff details
name = input("Enter Staff Name: ")
role = input("Enter Staff Role: ")

# Open webcam
cap = cv2.VideoCapture(0)

saved = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    faces = app.get(frame)

    for face in faces:

        # Get embedding
        embedding = face.embedding

        # Convert embedding to binary
        embedding_blob = embedding.tobytes()

        # Save into database
        cursor.execute("""
        INSERT INTO staff(name, role, embedding)
        VALUES (?, ?, ?)
        """, (name, role, embedding_blob))

        conn.commit()

        saved = True

        cv2.putText(frame, "Staff Registered",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

    cv2.imshow("Register Staff", frame)

    if saved:
        cv2.waitKey(2000)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
conn.close()
cv2.destroyAllWindows()

print("Staff registered successfully")