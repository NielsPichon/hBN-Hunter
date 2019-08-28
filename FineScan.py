##################################################################################################
# Automatic Fine Few layers hBN flakes detection 
#
#
# Copyright(c) 2019, Niels PICHON. All Rights Reserved 
# Permission to use, copy, modify, and distribute this software 
# and its documentation for educational, research, and not-for-profit purposes, 
# without fee and without a signed licensing agreement, is hereby granted, 
# provided that the above copyright notice, this paragraph and the following paragraphs 
# appear in all copies, modifications, and distributions.
# Contact author (niels.pichon@hotmail.fr) for commercial licensing opportunities.
#
# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE 
# AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, 
# IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, 
# SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
#
# hBN Thickness-Color Look up table credit : 
# Lene Gammelgaard, Technical University of Denmark, 
# Department of Physics, 
# Fysikvej, building 311, 
# 2800 Kgs. Lyngby
#
##################################################################################################


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
parser.add_argument('--n_clusters', default = 16, type=int, help='Number of color clusters to look for. Default = 16')
parser.add_argument('--folder', required = True, type=str, help='Folder where microscope image to detect suitable hBN flakes on are.')
parser.add_argument('--ClusteringSpace', default = 0, type=int, help='Color Space in which to perform the clustering. 1 for HSV, 0 for RGB. Default = 0 (RGB)')
parser.add_argument('--ThicknessInterpSpace', default = 1, type=int, help='Color Space in which to the thickness computation. 1 for HSV, 0 for RGB. Default = 1 (HSV)')
parser.add_argument('--minThickness', default = 31, type=float, help='Minimum acceptable thickness. Default = 31')
parser.add_argument('--maxThickness', default = 100, type=float, help='Maximum acceptable thickness. Default = 100')
parser.add_argument('--thicknessFilter', default = 3, type=float, help='Thickness threshold (in nm) under which clusters will be aggregated. 0 to turn off. Default = 3')
parser.add_argument('--minArea', default = 1000, type=int, help='Minimum acceptable area (in pixels) for a flake. Default = 1000')
parser.add_argument('--background', default = 'background.png', type = str, help = 'Picture of the background. Default = background.png')
parser.add_argument('--exportDir', default = './FoundFlakes', type = str, help = 'Path to the directory where found flakes will be saved. Default = ./FoundFlakes') 
parser.add_argument('--exportImg', default = 1, type = int, help = 'If 1, contours will be drawn on images and saved. (Slightly Slower). Default = 1')  
parser.add_argument('--stretch', type=str, default = "gray", help='Stretching method. Default = gray')
parser.add_argument('--gray', type=float, default = 0.8, help='Value of gray (in percentage of the compressed range), if gray strech method is used')
parser.add_argument('--noiseFilter', default = 0, type =int, help ='Strength of the noise filtering. Default = 0 (no filtering))')
parser.add_argument('--kernTolerance', default = 0.1, type =float, help ='Tolerance (in percentage) of the device fitting. Default = 0.05')
parser.add_argument('--minX', default = 40, type =int, help ='Min width size of device to fit. Default = 40px')
parser.add_argument('--minY', default = 80, type =int, help ='Min length size of device to fit. Default = 80px')
parser.add_argument('--HSVFilter', default = 10, type=float, help='Distance in the HSV spce under which clusters will be aggregated. Default = 10')


args = parser.parse_args()
print(args)
					
K = args.n_clusters
HSVClustering = args.ClusteringSpace
RGBDistance = 1 - args.ThicknessInterpSpace
minThickness = args.minThickness
maxThickness = args.maxThickness
thicknessFilter = args.thicknessFilter
minArea = args.minArea
folder = args.folder
backgroundPath = args.background
exportDir = args.exportDir
if (args.exportImg == 1):
	exportImg = True
else:
	exportImg = False
stretchType = args.stretch
grayPercentage = args.gray
noiseFilter = args.noiseFilter
kernelTolerance = args.kernTolerance
sizeX = args.minX
sizeY = args.minY
HSVFilter = args.HSVFilter
	
####################################################################################################
#gets the list of all microscope images in given folder
fileList = glob.glob("./" + folder + "/*.png") 

#register the bit-depth of the input images to export the background in the right bit depth
bitdepthTestImage = cv2.imread(fileList[0])
bitdepth = bitdepthTestImage.dtype
print('image bitdepth:', bitdepth)

###################################################################################################

#table of colors and associated thicknesses for a 90nm oxide on Silicon
colorTable = np.zeros((22,1,3), np.uint8)
colorTable[0,0,:] = (92,71,103)
colorTable[1,0,:] = (112,87,119)
colorTable[2,0,:] = (126,70,60)
colorTable[3,0,:] = (179,108,47)
colorTable[4,0,:] = (181,115,49)
colorTable[5,0,:] = (205,146,67)
colorTable[6,0,:] = (218,169,94)
colorTable[7,0,:] = (227,190,133)
colorTable[8,0,:] = (233,206,174)
colorTable[9,0,:] = (231,211,202)
colorTable[10,0,:] = (210,210,221)
colorTable[11,0,:] = (163,202,222)
colorTable[12,0,:] = (167,206,227)
colorTable[13,0,:] = (114,186,223)
colorTable[14,0,:] = (155,114,226)
colorTable[15,0,:] = (211,96,159)
colorTable[16,0,:] = (216,210,133)
colorTable[17,0,:] = (209,213,160)
colorTable[18,0,:] = (142,209,213)
colorTable[19,0,:] = (227,140,227)
colorTable[20,0,:] = (255,155,172)
colorTable[21,0,:] = (222,167,154)

colorTableHSV = cv2.cvtColor(colorTable, cv2.COLOR_BGR2HSV)

thicknessTable = [0, 0, 7, 14, 16, 22, 27, 31, 43, 60, 78, 90, 92, 100, 120, 138, 168, 196, 217, 275, 285, 289]

#records mean target background color
targetColor = np.zeros((1,1,3), np.uint8)
targetColor[0,0,:] = (102, 79, 111)
targetColorHSV = cv2.cvtColor(np.array(targetColor), cv2.COLOR_BGR2HSV)
###################################################################################################
def colorRangeCompression(image, stretchType, grayPercentage, bitdepth):
	#compresses the color range to drastically increase the contrast
	
	#convert to grayscale
	grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	#compute histogram
	hist = cv2.calcHist([grayImg], [0], None, [256], [0, 256])
	
	#Compute median value of histogram
	medianHist = np.median(hist, axis = 0)
	
	#find rightmost peak (i.e. first local maximum with value above median)
	windowSize = 25
	window_overlap = 10
	imin = 255 - windowSize + 1
	imax = 255
	locMax = 0
	locMaxIdx = 255
	while ((imin > 0) & (locMax < medianHist)):
		locMaxIdx = np.argmax(hist[imin : imax]) + imin
		if ((hist[locMaxIdx] > hist[locMaxIdx + 1]) & (hist[locMaxIdx] > hist[locMaxIdx + 1])) : 
			locMax = hist[locMaxIdx]
		else:
			locMax = 0
		
		imin -= window_overlap
		imax -= window_overlap
	
	rigthPeakPos = locMaxIdx	
	
	#find left Most Peak (i.e. first local maximum with value above median)
	windowSize = 25
	window_overlap = 10
	imin = 1
	imax = windowSize
	locMax = 0
	locMaxIdx = 1
	while ((imax < 255) & (locMax < medianHist)):
		locMaxIdx = np.argmax(hist[imin : imax]) + imin
		if ((hist[locMaxIdx] > hist[locMaxIdx + 1]) & (hist[locMaxIdx] > hist[locMaxIdx + 1])) : 
			locMax = hist[locMaxIdx]
		else:
			locMax = 0
		
		imin += window_overlap
		imax += window_overlap
	
	leftPeakPos = locMaxIdx	
	
	#compute left range bound
	leftBound = np.floor((rigthPeakPos+leftPeakPos)/2)
	
	if (stretchType == "log") : 
			#compress range (logarythmically) 
			slope = 255.0 / (np.log(np.float(rigthPeakPos)) - np.log(np.float(leftBound)))
			shift = - slope * np.log(np.float(leftBound))
			image.astype(np.float32)
			image = np.log(image) * slope + shift
	elif (stretchType == "exp") : 
		#compress range (exponentially) 
		slope = 1 / (np.exp(np.float(rigthPeakPos)/255.0) - np.exp(np.float(leftBound)/255.0))
		shift = - slope * np.exp(np.float(leftBound)/255.0)
		image.astype(np.float32)
		image = 255.0 * (np.exp(image/255.0) * slope + shift)
	elif (stretchType == "lin"):
		#compress range (linearly) 
		slope = 255.0 / (np.float(rigthPeakPos) - np.float(leftBound))
		shift = - slope * np.float(leftBound)
		image.astype(np.float32)
		image = image * slope + shift
	elif (stretchType == "gray"): 
		rigthPeakPos = np.float(rigthPeakPos)
		leftBound = np.float(leftBound)
		grayValue = grayPercentage * rigthPeakPos + (1 - grayPercentage) * leftBound
		
		slopeL = 127 / (np.float(grayValue) - np.float(leftBound))
		shiftL = - slopeL * np.float(leftBound)
		
		slopeR = 128 / (np.float(rigthPeakPos) - np.float(grayValue))
		shiftR = - slopeR * np.float(grayValue)
		
		imageL = np.clip(image, leftBound, grayValue)
		imageR = np.clip(image, grayValue, rigthPeakPos)
		image = (imageL * slopeL + shiftL) + (imageR * slopeR + shiftR)
	
	else : 
		print('Unknown stretching type. Program will stop.')
		exit()
		
	#cast back to original bit depth
	correctedImage = (np.clip(np.around(image),0,255)).astype(bitdepth)

	return correctedImage

				
######################################################################################################
#detects hBN flakes on a microscope image
def hBNDetector(img, resized_image, sizeRatio, K, HSVClustering, RGBDistance, minThickness, maxThickness, thicknessFilter, minArea, colorTableHSV, thicknessTable, exportImg):
	
	if (HSVClustering == 1):
		Z = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
	Z = resized_image.reshape((-1,3))

	# convert to np.float32
	Z = np.float32(Z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,labels,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	###################################################################################################
	#convert the mean colors to uint8
	center = np.uint8(center)

	#reshape the labels to the image size	
	labels = labels.reshape((resized_image.shape[:-1]))	

	#check weather the color is of interest or not and if yes, "measure" continuous crystal size
	dist_to_color = np.zeros(22, float)
	thickness = np.zeros(22, float)
	for i in range(0,K) :
		clusterColor = np.zeros((1,1,3), np.uint8) 
		clusterColor[0,0,0] = center[i][0]
		clusterColor[0,0,1] = center[i][1]
		clusterColor[0,0,2] = center[i][2]
		
		if ((HSVClustering == 0) & (RGBDistance == 0)):
			clusterColor = cv2.cvtColor(clusterColor, cv2.COLOR_BGR2HSV)
		
		minDist = 10000000000000
		minDistIdx = -1
		
		#compute HSV distance (one might want to disregard saturation)
		j = 0
		if (RGBDistance == 0):
			for crystalColor in colorTableHSV:
				dist_to_color[j] = sqrt(pow(float(crystalColor[0,0]) - float(clusterColor[0,0,0]), 2) + pow(float(crystalColor[0,1]) - float(clusterColor[0,0,1]), 2) + pow(float(crystalColor[0,1]) - float(clusterColor[0,0,1]), 2))
				if (dist_to_color[j] < minDist):
					minDist = dist_to_color[j]
					minDistIdx = j
				j += 1
		else:
			for crystalColor in colorTable:
				dist_to_color[j] = sqrt(pow(float(crystalColor[0,0]) - float(clusterColor[0,0,0]), 2) + pow(float(crystalColor[0,1]) - float(clusterColor[0,0,1]), 2) + pow(float(crystalColor[0,1]) - float(clusterColor[0,0,1]), 2))
				if (dist_to_color[j] < minDist):
					minDist = dist_to_color[j]
					minDistIdx = j
				j += 1
				
		#interpolate thickness based onn HSV distance
		if (minDistIdx == len(colorTableHSV) - 1):
			thickness[i] = (dist_to_color[minDistIdx -1] * thicknessTable[minDistIdx] + dist_to_color[minDistIdx] * thicknessTable[minDistIdx - 1]) / (dist_to_color[minDistIdx] + dist_to_color[minDistIdx - 1])
		elif (minDistIdx == 0):
			thickness[i] = (dist_to_color[1] * thicknessTable[0] + dist_to_color[0] * thicknessTable[1]) / (dist_to_color[1] + dist_to_color[0])
		elif (dist_to_color[minDistIdx - 1] < dist_to_color[minDistIdx + 1]):
			thickness[i] = (dist_to_color[minDistIdx - 1] * thicknessTable[minDistIdx] + dist_to_color[minDistIdx ] * thicknessTable[minDistIdx - 1]) / (dist_to_color[minDistIdx] + dist_to_color[minDistIdx - 1])
		else :
			thickness[i] = (dist_to_color[minDistIdx + 1] * thicknessTable[minDistIdx] + dist_to_color[minDistIdx] * thicknessTable[minDistIdx + 1]) / (dist_to_color[minDistIdx] + dist_to_color[minDistIdx + 1])
	###################################################################################################
	#merge clusters that are too close together
	ignoreCluster = np.zeros(K,int)
	height, width = labels.shape

	if(thicknessFilter > 0):
		for idx in range(0,K):
			min = 100000000
			minIdx = -1
			#find closest cluster in thickness
			for i in range(idx + 1, K):
				if (abs(thickness[i] - thickness[idx]) < min):
					min = abs(thickness[i] - thickness[idx])
					minIdx = i
			#if the coresponding cluster is below the threshold, merge both clusters
			if (min < thicknessFilter):
				#flag current cluster to be ignored
				ignoreCluster[idx] = 1
				#register the thickness as the min (completely arbitrary)
				thickness[minIdx] = np.minimum(thickness[idx], thickness[minIdx])
				#transfer the labels from one cluster to the other
				for i in range(0, height): 
					for j in range(0, width): 
						if (labels[i,j] == idx):
							labels[i,j] = minIdx						
	##################################################################################################
	#segment suitable layer thickness crystals to find right areas.
	flakesContours = []
	flakesArea = []
	flakesCentroid = []
	flakesThickness = []

	for i in range(0,K):
		
		if ((thickness[i] > minThickness) & (thickness[i] < maxThickness) & (ignoreCluster[i] == 0)):
			#if within suitable thickness range, create mask
			mask = cv2.inRange(labels, i, i)
					
			#detect contours
			contours, hierarchy  = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE )
			
			#for each detected contour, check if area if sufficient
			for cnt in contours:
				area = cv2.contourArea(cnt)
				if(area > minArea):
					M = cv2.moments(cnt)
					flakesArea.append(area)
					flakesCentroid.append([int(M['m10']/M['m00'] / sizeRatio), int(M['m01']/M['m00'] / sizeRatio)])
					flakesContours.append((np.around(cnt / sizeRatio)).astype(np.int))
					flakesThickness.append(thickness[i])
	###################################################################################################
	if (exportImg):
		img = cv2.drawContours(img, flakesContours, contourIdx = -1, color = (255,0,0), thickness = 2)
	# font = cv2.FONT_HERSHEY_SIMPLEX
	# for i in range(0,len(flakesThickness)):
		# cv2.putText(resized_image, str(int(flakesThickness[i])), (flakesCentroid[i][0], flakesCentroid[i][1]) , font, .5, (0,0,0), 1, cv2.LINE_AA)
	return flakesCentroid, flakesContours, img

######################################################################################################

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))
	
######################################################################################################
def drawRotateRectange(image, center, angle, width, height, color = (255,0, 0), thickness = 2):
	#rotation origin is with width horizontal. Draws a rotated rectangle on image because #@"# openCV for python doesn't have that function
	
	#convert angle to radians
	angle = angle / 180 * np.pi
	
	#define rectangle angles
	pt1 = [+ width / 2 , + height / 2]
	pt2 = [- width / 2 , + height / 2]
	pt3 = [- width / 2 , - height / 2]
	pt4 = [+ width / 2 , - height / 2]
	
	#rotate rectangle
	pt1 = np.floor([pt1[0] * np.cos(angle) + pt1[1] * np.sin(angle) , - pt1[0] * np.sin(angle) + pt1[1] * np.cos(angle)])
	pt2 = np.floor([pt2[0] * np.cos(angle) + pt2[1] * np.sin(angle) , - pt2[0] * np.sin(angle) + pt2[1] * np.cos(angle)])
	pt3 = np.floor([pt3[0] * np.cos(angle) + pt3[1] * np.sin(angle) , - pt3[0] * np.sin(angle) + pt3[1] * np.cos(angle)])
	pt4 = np.floor([pt4[0] * np.cos(angle) + pt4[1] * np.sin(angle) , - pt4[0] * np.sin(angle) + pt4[1] * np.cos(angle)])
	
	#center rectangle
	pt1 += center
	pt2 += center
	pt3 += center
	pt4 += center
	
	#convert to int
	pt1 = np.int64(pt1)
	pt2 = np.int64(pt2)
	pt3 = np.int64(pt3)
	pt4 = np.int64(pt4)
	
	image = cv2.line(image, (pt1[0],pt1[1]), (pt2[0],pt2[1]), color = color, thickness = thickness)
	image = cv2.line(image, (pt3[0],pt3[1]), (pt2[0],pt2[1]), color = color, thickness = thickness)
	image = cv2.line(image, (pt3[0],pt3[1]), (pt4[0],pt4[1]), color = color, thickness = thickness)
	image = cv2.line(image, (pt1[0],pt1[1]), (pt4[0],pt4[1]), color = color, thickness = thickness)
	
	return image
	
###################################################################################################
def deviceFitting(img, resized_image, flakeContours, sizeRatio, K, HSVClustering, HSVFilter, minArea, noiseFilter, kernelTolerance, sizeX, sizeY):
	#will try to fit a device in the flakes
	
	# denoise the image for better results (SLOW!!!!!!)
	if (noiseFilter > 0):
		resized_image = cv2.fastNlMeansDenoisingColored(resized_image, None, noiseFilter)
	
	if (HSVClustering == 1):
		Z = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
	Z = resized_image.reshape((-1,3))
	
	# convert to np.float32
	Z = np.float32(Z)
	
	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,labels,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	###################################################################################################
	#convert the mean colors to uint8
	center = np.uint8(center)
		
	#reshape the labels to the image size	
	labels = labels.reshape((resized_image.shape[:-1]))	
	
	#merge clusters that are too close together
	ignoreCluster = np.zeros(K,int)
	height, width = labels.shape
	
	if(HSVFilter > 0):
		for idx in range(0,K):
			clusterColor = np.zeros((1,1,3), np.uint8) 
			clusterColor[0,0,0] = center[idx][0]
			clusterColor[0,0,1] = center[idx][1]
			clusterColor[0,0,2] = center[idx][2]
			
			#remove background if black
			if ((clusterColor[0,0,:] == (0,0,0)).all()):
				ignoreCluster[idx] = 1
			else:
				#if we weren't HSV space already, convert to HSV space
				if (HSVClustering != 1):
					clusterColor = cv2.cvtColor(clusterColor, cv2.COLOR_BGR2HSV)
				
				min = 100000000
				minIdx = -1 
				#find closest cluster in HSV space
				for i in range(idx + 1, K):
					clusterColor2 = np.zeros((1,1,3), np.uint8) 
					clusterColor2[0,0,0] = center[i][0]
					clusterColor2[0,0,1] = center[i][1]
					clusterColor2[0,0,2] = center[i][2]
					#if we weren't HSV space already, convert to HSV space
					if (HSVClustering != 1):
						clusterColor2 = cv2.cvtColor(clusterColor2, cv2.COLOR_BGR2HSV)
						
					HSVdist = np.max(abs(clusterColor - clusterColor2))
					
					if (HSVdist < min):
						min = HSVdist
						minIdx = i
						
				#if the coresponding cluster is below the threshold, merge both clusters
				if (min < HSVFilter):
					#flag current cluster to be ignored
					ignoreCluster[idx] = 1
					center[idx] = center[minIdx]
					#transfer the labels from one cluster to the other
					for i in range(0, height): 
						for j in range(0, width): 
							if (labels[i,j] == idx):
								labels[i,j] = minIdx
	#print clustered image for debug
	res = center[labels.flatten()]
	res2 = res.reshape((resized_image.shape))
	###################################################################################################	
	#segment suitable layer thickness crystals to find right areas.
	flakesContours = []
	flakesArea = []
	flakesCentroid = []
	flakesThickness = []
	
	for i in range(0,K):
		
		if (ignoreCluster[i] == 0):
			#create mask
			mask = cv2.inRange(labels, i, i)
			
			#set cluster to be ignored in further processing 
			ignoreCluster[i] = 1
			
			#detect contours
			contours, hierarchy  = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE )
			
			for cnt in contours:
				area = cv2.contourArea(cnt)
				if(area > minArea):
					M = cv2.moments(cnt)
					flakesArea.append(area)
					flakesCentroid.append([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])
					flakesContours.append(cnt)
					#set cluster to be considered for further processing
					ignoreCluster[i] = 0		
	##################################################################################################
	#filter with given kernel to find if such patch exists in the picture	
	kernelSize = sizeX * sizeY
	objective = kernelSize * (1 - kernelTolerance)
	
	#create kernel of min device size and shape (rectangle only)
	kernel = np.ones((sizeX, sizeY),dtype=np.uint8)
	
	device = []
	deviceRot = []
	
	for j in range(0,K):
		if (ignoreCluster[j] == 0):
			#create mask
			mask = cv2.inRange(labels, j, j)
			# make sure the masked part is 1 (not 255)
			mask[mask>0] = 1
				
			for i in range (0,180):
				#rotate kernel
				k = rotate_bound(kernel, i)
				#filter image with kernel			
				filterIm = cv2.filter2D(src = mask, ddepth = cv2.CV_16S, kernel = k, borderType = cv2.BORDER_CONSTANT )
								
				#look for maximum and check if equals to kernel pixels (+/- tolerance for rotation artifacts)			
				max = np.amax(filterIm)
				maxX, maxY = np.unravel_index(np.argmax(filterIm, axis=None), filterIm.shape)
							
				if (max >= objective):
					buffer = np.zeros(mask.shape)
					buffer[filterIm==max] = 255
					
					inside = False
					for flake in flakeContours : 
						dist = cv2.pointPolygonTest(flake, (int(maxY / sizeRatio), int(maxX / sizeRatio)),  False)
						if (dist > 0) :
							inside = True
							break
					if (inside)	:			
						device.append([int(maxY / sizeRatio), int(maxX / sizeRatio)])
						deviceRot.append(i)								
						break
	#####################################################################################################
	# #print fitted devices
	idx = -1
	for dev in device:
		idx += 1
		img = drawRotateRectange(image = img, center = dev, angle = deviceRot[idx], width = sizeX / sizeRatio, height = sizeY /sizeRatio)
	
	return device, deviceRot, img
#############################################################################################################
#############################################################################################################
start_time = time.time()

#check if output directory exists. Else create it
if (not os.path.isdir(exportDir)):
	os.makedirs(exportDir)

idx = 0
framesWithFlakes = []
flakes = []
flakeNb = 0
#for each image in image folder 
for file in fileList :

	idx = idx + 1
	print('Detecting hBN on image %s of %s' %(idx, len(fileList)))
	
	#load image
	img = cv2.imread(file)
	
	height, width, _ = img.shape
	ratio = 500 / width
	resizedImg = cv2.resize(img, (500, int(height * ratio))) 
	
	# cv2.imshow('', resizedImg)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	
	#perform color based detection
	flakesCentroid, flakesContours, imgWithContour = hBNDetector(img, resizedImg, ratio, K, HSVClustering, RGBDistance, minThickness, maxThickness, thicknessFilter, minArea, colorTableHSV, thicknessTable, exportImg)
	
	#if at least one crystal candidate, compress colors and check where one can put a device.
	if (len(flakesCentroid) > 0):
		compressedImg = colorRangeCompression(resizedImg, stretchType, grayPercentage, bitdepth)
		deviceCenters, deviceRot, img = deviceFitting(img, resizedImg, flakesContours, ratio, K, HSVClustering, HSVFilter, minArea, noiseFilter, kernelTolerance, sizeX, sizeY)
	
	if (len(deviceCenters) > 0) : 
		flakes.append(flakesCentroid)
		framesWithFlakes.append(file)
		if (exportImg):
			_, tail = os.path.split(file)
			cv2.imwrite(exportDir + '/' + tail, img)
		flakeNb += len(flakesCentroid)

print('Total detection completed in %s seconds' %(time.time() - start_time))
print('%s flake candidates detected' %flakeNb)	
				
summary = open(exportDir + '/FoundFlakes.txt', 'w')
summary.write('Detection performed in %s seconds \n \n' %(time.time() - start_time))
summary.write('List of pictures containing flakes (total = %s flakes): \n' %flakeNb)
for frame in framesWithFlakes:
	_, tail = os.path.split(frame)
	summary.write(tail + '\n')
summary.close()					
print('Summary exported to ' + exportDir + '/FoundFlakes.txt')