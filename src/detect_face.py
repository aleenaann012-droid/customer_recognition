import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis()
app.prepare(ctx_id=-1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    faces = app.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        age = face.age
        gender = face.gender

        # Convert gender number to text
        gender_text = "Male" if gender == 1 else "Female"

        # Create display text
        text = f"Age: {age} Gender: {gender_text}"
        
        cv2.rectangle(
            frame,
            (box[0], box[1]),
            (box[2], box[3]),
            (0,255,0),
            2
        )
        cv2.putText(
            frame,
            text,
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()