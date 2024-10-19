import cv2
from ultralytics import solutions
import threading

def reset(self):
	self.classwise_counts = {}

solutions.ObjectCounter.reset = reset

class Monitor:
	def __init__(self, intersection_id, direction):
		self.intersection_id = intersection_id
		self.direction = direction

	def start(self):
		# 2: car, 3: motorcycle, 5: bus, 7: truck 
		self.count_specific_classes("assets/test1.mp4", "output_specific_classes.avi", "yolo11n.pt", [2, 3, 5, 7])

	def count_specific_classes(self, video_path, output_video_path, model_path, classes_to_count):
		cap = cv2.VideoCapture(video_path)
		assert cap.isOpened(), "Error reading video file"
		w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
		video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

		FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		line_points = [(0, FRAME_HEIGHT - 200), (FRAME_WIDTH, FRAME_HEIGHT - 200), (FRAME_WIDTH, FRAME_HEIGHT - 100), (0, FRAME_HEIGHT - 100)]
		counter = solutions.ObjectCounter(show=True, region=line_points, model=model_path, classes=classes_to_count)

		def update():
			print('UPDATE ', counter.classwise_counts)
			counter.reset()
			threading.Timer(3, update).start()
		
		threading.Timer(3, update).start()

		while cap.isOpened():	
			success, im0 = cap.read()
			if not success:
				print("Video frame is empty or video processing has been successfully completed.")
				break
			im0 = counter.count(im0)
			video_writer.write(im0)

		cap.release()
		video_writer.release()
		cv2.destroyAllWindows()
	