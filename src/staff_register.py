import cv2
import sqlite3
import numpy as np
from insightface.app import FaceAnalysis
import threading
# ==========================================
# LOAD INSIGHTFACE MODEL
# ==========================================

app = FaceAnalysis(name='buffalo_l')

# CPU MODE
app.prepare(ctx_id=-1)
# ==========================================
# SHARED VARIABLES
# ==========================================

frame = None
display_frame = None
detected_faces = []

# ==========================================
# CONNECT DATABASE
# ==========================================

conn = sqlite3.connect("database/customers.db")
cursor = conn.cursor()

# ==========================================
# INPUT STAFF DETAILS
# ==========================================

name = input("Enter Staff Name: ")
role = input("Enter Staff Role: ")
cursor.execute("""
SELECT * FROM staff
WHERE name = ?
""", (name,))

existing_staff = cursor.fetchone()

if existing_staff:

    print("Staff already exists")

    conn.close()

    exit()

# ==========================================
# CHECK IF STAFF ALREADY EXISTS
# ==========================================

cursor.execute("""
SELECT * FROM staff
WHERE name = ?
""", (name,))

existing_staff = cursor.fetchone()

if existing_staff:

    print("Staff already registered")

else:

    # ==========================================
    # OPEN WEBCAM
    # ==========================================

    cap = cv2.VideoCapture(0)

    saved = False

    print("Look at camera to register...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        faces = app.get(frame)

        for face in faces:

            # Get embedding
            embedding = face.embedding.astype(np.float32)

            # Convert embedding to binary
            embedding_blob = embedding.tobytes()

            # Save into database
            cursor.execute("""
            INSERT INTO staff(name, role, embedding)
            VALUES (?, ?, ?)
            """, (name, role, embedding_blob))

            conn.commit()

            print("Staff registered successfully")

            saved = True

            break

        cv2.imshow("Register Staff", frame)

        # Exit after successful save
        if saved:
            cv2.waitKey(2000)
            break

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# CLOSE DATABASE
# ==========================================

conn.close()