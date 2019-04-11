'''
Shape Detector

Authors:
Revan MacQueen
Jordan McCarthy

This program handles the logic of shape detection
'''

# import the necessary packages
import cv2
 
class ShapeDetector:
	def __init__(self):
		pass
 
	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, .04 * peri, True)
		if cv2.contourArea(c) > 4000:
			#if the shape is a triangle, it will have 3 vertices
			if len(approx) == 3:
				shape = "triangle"
	 
			# if the shape has 4 vertices, it is either a square or
			# a rectangle
			elif len(approx) == 4:
				# compute the bounding box of the contour and use the
			
				# a square will have an aspect ratio that is approximately
				# equal to one, otherwise, the shape is a rectangle
				shape = "square"
	 
			# otherwise, we assume the shape is a circle
			else:
				shape = "circle"
	 
			# return the name of the shape
			return shape