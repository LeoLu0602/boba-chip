import cv2
from ultralytics import solutions
import threading

def reset(self):
    self.classwise_counts = {}

solutions.ObjectCounter.reset = reset

class Monitor:
    def __init__(self, intersection_id, direction, T):
        self.intersection_id = intersection_id
        self.direction = direction  # 1 or 2
        self.T = T
        self.is_active = False
        self.traffic = 0

    def get_is_active(self):
        return self.is_active

    def get_traffic(self):
        return self.traffic

    def set_is_active(self, val):
        self.is_active = val    

    def start(self):
        self.is_active = True
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        self.count_specific_classes(
            "assets/test1.mp4",
            "output_specific_classes.avi",
            "yolo11n.pt",
            [2, 3, 5, 7],
        )

    def count_specific_classes(
        self, video_path, output_video_path, model_path, classes_to_count
    ):
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (
            int(cap.get(x))
            for x in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
            )
        )
        video_writer = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        counting_region = [
            (0, FRAME_HEIGHT - 200),
            (FRAME_WIDTH, FRAME_HEIGHT - 200),
            (FRAME_WIDTH, FRAME_HEIGHT - 100),
            (0, FRAME_HEIGHT - 100),
        ]
        counter = solutions.ObjectCounter(
            show=True, region=counting_region, model=model_path, classes=classes_to_count
        )

        def update():
            traffic = 0

            if "car" in counter.classwise_counts:
                traffic += counter.classwise_counts["car"]["IN"] * 2

            if "motorcycle" in counter.classwise_counts:
                traffic += counter.classwise_counts["motorcycle"]["IN"] * 1

            if "bus" in counter.classwise_counts:
                traffic += counter.classwise_counts["bus"]["IN"] * 30

            if "truck" in counter.classwise_counts:
                traffic += counter.classwise_counts["truck"]["IN"] * 1

            traffic /= self.T
            self.traffic = traffic
            counter.reset()

            if self.is_active:
                threading.Timer(self.T, update).start()
            

        threading.Timer(self.T, update).start()

        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print(
                    "Video frame is empty or video processing has been successfully completed."
                )
                break

            im0 = counter.count(im0)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        self.set_is_active(False)
