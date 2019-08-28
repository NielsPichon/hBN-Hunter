import numpy as np
import cv2
import sys
import time
import argparse
from math import pow, sqrt
import glob
from sys import getsizeof
import os

###################################################################################################
#Parsing Arguments
parser = argparse.ArgumentParser(description='Finds hBN flakes using color clustering')
parser.add_argument('--image', required = True, type=str, help='Folder where microscope image to detect suitable hBN flakes on are.')
parser.add_argument('--back', required = True, type = str, help='background image')
parser.add_argument('--distThresh', default = 2, type = int , help= 'gray distance') 
parser.add_argument('--thresh', default = 0.01, type = float, help = 'activity threshold')
parser.add_argument('--minArea', default = 100, type = float, help = 'activity threshold')

args = parser.parse_args()
print(args)

im = cv2.imread(args.image)
back = cv2.imread(args.back)
thresh = args.thresh
distThresh = args.distThresh
minArea = args.minArea


height, width, channels = im.shape
size = height * width

#substract background
buffer = cv2.absdiff(im, back)

#threshold to get only things that stand out from the background
buffer = cv2.cvtColor(buffer, cv2.COLOR_BGR2GRAY)
_,buffer = cv2.threshold(buffer, distThresh, 255, cv2.THRESH_BINARY)

cv2.imshow('buff', buffer)
cv2.waitKey(0)
cv2.destroyAllWindows()	

erodeNb = np.int(np.sqrt(minArea) / 2)
kernel = np.ones((2,2),np.uint8)
buffer = cv2.erode(buffer,kernel,iterations = erodeNb)

cv2.imshow('buff', buffer)
cv2.waitKey(0)
cv2.destroyAllWindows()	

activityCounter  = cv2.countNonZero(buffer)
if (activityCounter > thresh * size):
	print('Active')
else :
	print('dead')