from cv2 import Mat
from imutils.video import VideoStream
import datetime
import argparse
import imutils
import time
import cv2
import numpy as np
import math
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--picamera", type=int, default=-1,
# 	help="whether or not the Raspberry Pi camera should be used")
# args = vars(ap.parse_args())
# # initialize the video stream and allow the cammera sensor to warmup
# vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
# time.sleep(2.0)

# # loop over the frames from the video stream
# while True:
# 	# grab the frame from the threaded video stream and resize it
# 	# to have a maximum width of 400 pixels
# 	frame = vs.read()
# 	frame = imutils.resize(frame, width=400)
# 	# show the frame
# 	cv2.imshow("Frame", frame)
# 	key = cv2.waitKey(1) & 0xFF
# 	# if the `q` key was pressed, break from the loop
# 	if key == ord("q"):
# 		break
# # do a bit of cleanup
# cv2.destroyAllWindows()
# vs.stop()

# Draws lines of given thickness over an image
def draw_lines(image, lines, thickness): 
   
    print(lines)
    line_image = np.zeros_like(image)
    color=[0, 0, 255]
    
    if lines is not None: 
        for x1, y1, x2, y2 in lines:
                    cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow("LINES", line_image)
    # Merge the image with drawn lines onto the original.
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)
    
    return combined_image
def get_coordinates (image, params):
     
    slope, intercept = params 
    y1 = image.shape[0]     
    y2 = int(y1 * (3/5)) # Setting y2 at 3/5th from y1
    x1 = int((y1 - intercept) / slope) # Deriving from y = mx + c
    x2 = int((y2 - intercept) / slope) 
    
    return np.array([x1, y1, x2, y2])

#Returns edges detected in an image
def canny_edge_detector(frame: Mat) -> Mat:
    
    # Convert to grayscale as only image intensity needed for gradients
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 5x5 gaussian blur to reduce noise 
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv2.Canny(blur, 50, 150)
    
    return canny 

# Returns averaged lines on left and right sides of the image
def avg_lines(image: Mat, lines: list): 
    
    left = [] 
    right = [] 
    
    for line in lines: 
        x1, y1, x2, y2 = line.reshape(4)

        # Fit polynomial, find intercept and slope 
        params = np.polyfit((x1, x2), (y1, y2), 1)  
        slope = params[0] 
        y_intercept = params[1] 
        
        if slope < 0: 
            left.append((slope, y_intercept)) #Negative slope = left lane
        else: 
            right.append((slope, y_intercept)) #Positive slope = right lane
    
    # Avg over all values for a single slope and y-intercept value for each line
    
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    
    # Find x1, y1, x2, y2 coordinates for left & right lines
    left_line = get_coordinates(image, left_avg) 
    right_line = get_coordinates(image, right_avg)
    
    return np.array([left_line, right_line])

def pipeline(img: Mat):

	## convert to hsv
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	## mask of blue
	mask1 = cv2.inRange(hsv, (100, 50, 70), (128, 255, 255))

	## mask o yellow
	mask2 = cv2.inRange(hsv, (25, 50, 70), (35, 255, 255))

	## final mask and masked
	mask = cv2.bitwise_or(mask1, mask2)
	masked = cv2.bitwise_and(img,img, mask=mask)

	cv2.imshow("mask", masked)
	canny_edges = canny_edge_detector(masked)
	cv2.imshow("canny", canny_edges)
	#Hough transform to detect lanes from the detected edges
	lines = cv2.HoughLinesP(
		canny_edges,
		rho=2,              #Distance resolution in pixels
		theta=np.pi / 180,  #Angle resolution in radians
		threshold=100,      #Min. number of intersecting points to detect a line  
		lines=np.array([]), #Vector to return start and end points of the lines indicated by [x1, y1, x2, y2] 
		minLineLength=40,   #Line segments shorter than this are rejected
		maxLineGap=5       #Max gap allowed between points on the same line
	)

	# Visualisations
	averaged_lines = avg_lines (img, lines)              #Average the Hough lines as left or right lanes
	combined_image = draw_lines(img, averaged_lines, 5)  #Combine the averaged lines on the real frame

	cv2.imshow("line_image", combined_image)
	key = cv2.waitKey(0) & 0xFF

img = cv2.imread("curve.jpg")

pipeline(img)