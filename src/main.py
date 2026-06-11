import cv2
import threading
import queue

from detection import detection_thread
from recognition import recognition_thread
import state

def main():
    threading.Thread(
        target=detection_thread,
        daemon=True
    ).start()

    threading.Thread(
        target=recognition_thread,
        daemon=True
    ).start()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        display = frame.copy()

        try:    
            if state.frame_queue.full():
                state.frame_queue.get_nowait()
            state.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

        with state.results_lock:
            current_results = state.draw_results.copy()

        for result in current_results:
            box = result["box"]
            text = result["text"]

            cv2.rectangle(
                display,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                text,
                (int(box[0]), max(10, int(box[1]) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("Recognition System", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()