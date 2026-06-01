import cv2

def create_tracker():

    try:
        return cv2.legacy.TrackerMOSSE_create()
    except:
        return cv2.TrackerMOSSE_create()