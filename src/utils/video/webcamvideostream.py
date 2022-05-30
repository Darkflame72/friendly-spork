# import the necessary packages
from threading import Thread
import cv2

from src.utils.video.stream import Stream

class WebcamVideoStream(Stream):
	def __init__(self, src=0, name="WebcamVideoStream", videocapture=None):
		# initialize the video camera stream and read the first frame
		# from the stream
		if videocapture is None:
			self.stream = cv2.VideoCapture(src)
		else:
			self.stream = videocapture
		(self.grabbed, self.frame) = self.stream.read()

		# initialize the thread name
		self.name = name

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		self._thread = Thread(target=self.update, name=self.name, args=())
		self._thread.daemon = True
		self._thread.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				self.stream.release()
				self._thread.join()
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
