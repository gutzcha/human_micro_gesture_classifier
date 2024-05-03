import random

import cv2
import torch
import time
import threading
from random import random
import os.path as osp

from mpigroup.load_model_inference import ModelInference, get_args
class VideoPipeline:
    def __init__(self, overlap_size=8, cycle_time=1, inference_function=None):
        self.last_display_time = 0
        self.buffer = []
        self.overlap_size = overlap_size
        self.cycle_time = cycle_time
        self.inference_function = inference_function
        self.frame_rate = 32  # fps of captured video
        self.display_frame_rate = 16  # fps for displaying frames
        self.display_lag = 0.25  # Display lag in seconds
        self.last_inference_time = time.time()
        self.inference_thread = threading.Thread(target=self._run_inference)
        self.inference_thread_stop_flag = threading.Event()  # Event flag to stop the thread
        self.stop_flag = False
        self.predictions = None
    def capture_video(self, source):


        # Capture video and add frames to buffer
        start_thread = True
        if source is None:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(source)
        while True:
            # Start the inference thread
            if len(self.buffer) >= 16 + self.overlap_size and start_thread:
                print('Starting thread')
                self.inference_thread.start()
                start_thread = False

            ret, frame = cap.read()
            if not ret:
                break
            self.buffer.insert(0, frame)
            self._manage_buffer()
            self._display_frames()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.stop_flag = True
        self.inference_thread_stop_flag.set()
        # Wait for the inference thread to finish
        self.inference_thread.join()

    def _manage_buffer(self):
        # Maintain buffer size
        if len(self.buffer) > 16 + self.overlap_size:
            self.buffer = self.buffer[:16]

        # Duplicate frames if buffer size is less than 16
        while len(self.buffer) < 16:
            self.buffer.append(self.buffer[-1])

    def _run_inference(self):
        try:
            while not self.stop_flag:
                # Run inference every cycle_time seconds
                if time.time() - self.last_inference_time >= self.cycle_time:
                    frames_to_infer = self.buffer[:16]
                    predictions = self.inference_function(frames_to_infer)
                    self.predictions = predictions
                    self.last_inference_time = time.time()
        except Exception as e:
            print(e)
            self.stop_flag = True
            self.inference_thread_stop_flag.set()
            self.inference_thread.join()
            self.predictions = None


    def _draw_predictions(self, frame,  predictions):
        # Draw predictions on frames
        if self.predictions is None:
            return
        threshold = 0.25
        if 'Legs_crossed' in predictions:
            del predictions['Legs_crossed']
        # if any(value > threshold for value in predictions.values()):

        for ind, (label, prediction) in enumerate(predictions.items()):
            if prediction > threshold:
                color = (0, 255, 0)
            else:
                color = (0, 0, 0)
            cv2.putText(frame, f"{label}: {prediction:.2f}", (10, 30+30*ind), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def _display_frames(self):
        # Display frames with a lag
        display_delay = 1 / self.display_frame_rate
        current_time = time.time()
        if current_time - self.last_display_time >= display_delay + self.display_lag:
            frame = self.buffer[-1]
            self._draw_predictions(frame, self.predictions)
            cv2.imshow('Frame', frame)
            self.last_display_time = current_time


def dummy_inference_function(frames):
    # Dummy inference function, replace with your actual implementation
    time.sleep(0.4)  # Simulating long inference time
    v1 = random()
    v2 = random()
    print('Running inference function')
    return {"Label1": v1, "Label2": v2}



if __name__ == "__main__":
    config_path = osp.join('..', 'model_configs', 'mpigroup_multiclass_inference_debug.yaml')
    args = get_args(config_path)
    inference_object = ModelInference(args)
    # path_to_video = "D:\\Project-mpg microgesture\\imigue\\0001.mp4"
    path_to_video = None
    pipeline = VideoPipeline(overlap_size=8, cycle_time=1, inference_function=inference_object.run_inference)
    pipeline.capture_video(source=path_to_video)
