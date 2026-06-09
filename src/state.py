# state.py

import queue
import threading

frame_queue = queue.Queue(maxsize=10)

det_queue = queue.Queue(maxsize=10)

draw_results = []

results_lock = threading.Lock()

new_recognitions_flag = False

frame_counter = 0

staff_presence = {}

recognized_tracks = {}

saved_tracks = set()