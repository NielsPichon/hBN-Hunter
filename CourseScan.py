##################################################################################################
# Automatic Few layers hBN flakes detection 
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
parser.add_argument('--minArea', default = 1500, type=int, help='Minimum acceptable area (in pixels) for a flake. Default = 1000')
parser.add_argument('--memory', default = 10**9, type=int, help='Max memory usage for the buffer in Bytes. Default = 10**9 Bytes')
parser.add_argument('--background', default = 'background.png', type = str, help = 'Picture of the background. Default = background.png')
parser.add_argument('--exportDir', default = './FoundFlakes', type = str, help = 'Path to the directory where found flakes will be saved. Default = ./FoundFlakes') 
parser.add_argument('--distThreshold', default = 2, type = int, help = 'Minimum distance from background in values of gray (8bit) in image activity detection. Default = 2')
parser.add_argument('--percentage', default = 0.7, type = float, help = 'Fraction of the image dimension above which chances are the flake is actually a border. Default = 0.7')
parser.add_argument('--ultraCourse', default = 0, type = int, help = 'Should ultra course detection be used ? If true (=1), faster but a lost of false detection. Default = 0')
parser.add_argument('--exportImg', default = 1, type = int, help = 'If 1, contours will be drawn on images and saved. (Slightly Slower). Default = 1')  
parser.add_argument('--createFullChip', default = 0, type = int, help = 'If 1, an image of the entire chip with the position of the found flakes will be created (SLOW). Default = 0')
parser.add_argument('--overlap', default = .1, type = float, help= 'In the case where the full chip image is to be created, sets the overlap that was used when capturing frames. Default = 0.1')

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
maxMemory = args.memory
backgroundPath = args.background
exportDir = args.exportDir
distThreshold = args.distThreshold
percentage = args.percentage
if (args.ultraCourse == 1):
	ultra = True
else:
	ultra = False
	
if (args.exportImg == 1):
	exportImg = True
else:
	exportImg = False
	
if (args.createFullChip == 1):
	fullChip = True
else:
	fullChip = False
overlap = args.overlap

	
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

#Extracts the background from a given collection of microscope images
def backgroundCorrection(fileList, maxMemory, bitdepth):
	start_time = time.time()
	#loads all pics from folder and stacks them in a buffer.
	buffer = []
	for i in range(0, len(fileList)):
		tmp = cv2.imread(fileList[i])
		buffer.append(tmp)
		if (getsizeof(buffer) > maxMemory): #limits the buffer size to 1 GB in memory to avoid memory overload
			break
	print('Buffer shape:', np.array(buffer).shape)
	print('Buffer loaded in %s seconds' %(time.time()- start_time))

	#computes median of the buffer
	start_time = time.time()
	background = np.median(buffer, axis = 0)
	print('Background extracted in %s seconds' %(time.time()- start_time))

	#convert back to the right bit depth
	background = background.astype(bitdepth)
	
	return background

###################################################################################################
def activityFilter(image, background, minFlakeArea, distThreshold = 2):
	#checks quickly if image is worth analyzing by counting number of pixels that are big enougth clusters of significantly different color from the background
	buffer  = activityMask(image, background, distThreshold )
		
	#detect contours	
	contours, hierarchy  = cv2.findContours(buffer.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE )
	
	#check if contours are big enough for being considered
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if(area > minFlakeArea):
			return True
			
	return False
###################################################################################################
def activityMask(img, background, distThreshold = 2):
	#creates a mask of where there is activity
	height, width, channels = img.shape
	size = height * width

	#substract background
	buffer = cv2.absdiff(img, background)
	
	#threshold to get only things that stand out from the background
	buffer = cv2.cvtColor(buffer, cv2.COLOR_BGR2GRAY)
	_,buffer = cv2.threshold(buffer, distThreshold, 255, cv2.THRESH_BINARY)
	
	#erode once to clean noise
	kernel = np.ones((2,2),np.uint8)
	buffer = cv2.erode(buffer,kernel,iterations = 5)
	
	return buffer
###################################################################################################
def AoIdetection(image, background, minFlakeArea, distThreshold = 2):
	buffer  = activityMask(image, background, distThreshold )

	#detect contours	
	contours, hierarchy  = cv2.findContours(buffer.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE )
	
	#check if contours are big enough for being considered
	candidates = []
	corners = []
	candidatesBackground = []
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if(area > minArea):
			x,y,w,h = cv2.boundingRect(cnt)
			subImg = image[y:y + h, x:x + w]
			candidates.append(subImg)
			corners.append([x, y])
			candidatesBackground.append(background[y:y + h, x:x + w])
	
	return candidates, corners, candidatesBackground
	
###################################################################################################
	
#corrects images for uneven illumination 
def illuminationCorrection(image, background, bitdepth):
	#computing the mean color of the background
	meanBackgroundColor = (np.mean(background.astype(np.float32), axis = 0))
	meanBackgroundColor = np.mean(meanBackgroundColor, axis = 0)

	#convert to float to avoid truncating
	image = image.astype(np.float32)
	background = background.astype(np.float32)

	#correct image
	correctedImage = np.divide(image, background) * meanBackgroundColor

	#cast back to original bit depth
	correctedImage = (np.clip(np.around(correctedImage),0,255)).astype(bitdepth)
	
	return correctedImage
		
###################################################################################################
		
#corrects colors to match those of the Color LUT
def colorCorrection(image, background, target):
	#computing the mean color of the background
	meanBackgroundColor = np.mean(background, axis = 0)
	meanBackgroundColor = np.mean(meanBackgroundColor, axis = 0)
	
	#converting to HSV space
	mean = np.zeros((1,1,3), np.uint8)
	mean[0,0,:] = meanBackgroundColor
	meanBackgroundColorHSV = cv2.cvtColor(np.array(mean), cv2.COLOR_BGR2HSV)

	#compute distance between current background and color target
	distHSV = meanBackgroundColorHSV - targetColorHSV

	#correct image
	imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	correctedImageHSV = imgHSV - distHSV
	correctedImage = cv2.cvtColor(correctedImageHSV, cv2.COLOR_HSV2BGR)
	
	return correctedImage
	
######################################################################################################
def borderFilter(contour, image):

	height, width, _ = image.shape
	
	#check if image corners are part of the contour
	pt1 = cv2.pointPolygonTest(contour,(0,0),True)
	pt2 = cv2.pointPolygonTest(contour,(width - 1, 0),True)
	pt3 = cv2.pointPolygonTest(contour,(0, height - 1),True)
	pt4 = cv2.pointPolygonTest(contour,(width - 1, height - 1),True)
	
	if (((abs(pt1)<=2) & (abs(pt2)<=2)) | ((abs(pt1)<=2) & (abs(pt3)<=2)) | ((abs(pt2)<=2) & (abs(pt4)<=2)) | ((abs(pt3)<=2) & (abs(pt4)<=2))):
		return False
	else:
		return True
		
######################################################################################################
def sizeFilter(subImg, image, percentage):

	hImg, wImg, _ = image.shape
	hSub, wSub, _ = subImg.shape
	
	if ((hSub > hImg * percentage) | (wSub > wImg * percentage)):
		return False
	else:
		return True		
		
######################################################################################################
#detects hBN flakes on a microscope image
def hBNDetector(image, K, HSVClustering, RGBDistance, minThickness, maxThickness, thicknessFilter, minArea, colorTableHSV, thicknessTable, background, targetColorHSV, percentage, exportImg, distThreshold = 2):
	
	height, width, channels = image.shape
	
	#detect flakes
	candidates,corners,candidatesBackground = AoIdetection(image, backgroundImg, minArea, distThreshold)
	
	flakesContours = []
	flakesArea = []
	flakesCentroid = []
	flakesThickness = []
	
	candidateIdx = -1
	
	for img in candidates:
		candidateIdx +=1
		if (sizeFilter(img, image, percentage)):
			
			#correct illumination
			img = illuminationCorrection(img, candidatesBackground[candidateIdx], bitdepth)
			
			#correct colors
			img = colorCorrection(img, backgroundImg, targetColorHSV)
			
			if (HSVClustering == 1):
				Z = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			Z = img.reshape((-1,3))

			# convert to np.float32
			Z = np.float32(Z)

			# define criteria, number of clusters(K) and apply kmeans()
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
			ret,labels,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
			###################################################################################################
			#convert the mean colors to uint8
			center = np.uint8(center)

			#reshape the labels to the image size	
			labels = labels.reshape((img.shape[:-1]))	

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
							if(borderFilter(cnt,img)):
								M = cv2.moments(cnt)
								flakesArea.append(area)
								x = int(M['m10']/M['m00']) + corners[candidateIdx][0]
								y = int(M['m01']/M['m00']) + corners[candidateIdx][1]
								flakesCentroid.append([x,y])
								flakesContours.append(cnt + corners[candidateIdx]) 
								flakesThickness.append(thickness[i])
		###################################################################################################
	if (exportImg):
		image = cv2.drawContours(image, flakesContours, contourIdx = -1, color = (255,0,0), thickness = 2)
			
	return flakesCentroid, image

######################################################################################################
#detects hBN flakes on a microscope image
def ultraCourseDetector(image, backgroundImg, minArea, exportImg, distThreshold = 2):
	
	height, width, channels = image.shape
	#detect flakes
	buffer  = activityMask(image, backgroundImg, distThreshold)

	#detect contours	
	contours, hierarchy  = cv2.findContours(buffer.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE )
	
	#check if contours are big enough for being considered
	candidates = []
	centers = []
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if(area > minArea):
			M = cv2.moments(cnt)
			candidates.append(cnt)
			centers.append([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])
	
	if (exportImg):	
		image = cv2.drawContours(image, candidates, contourIdx = -1, color = (255,0,0), thickness = 2)

	return centers, image

######################################################################################################
def getXYframePosition(file):
	#extracts x and y position from the name of the image
	
	_, tail = os.path.split(file)
	l = list(filter(str.isdigit, tail))
	index_x = ''
	index_y = ''
	
	for i in range(0, int(len(l)/2)):
		index_x = index_x + l[i]
		index_y = index_y + l[i + int(len(l)/2)]
	
	x = int(index_x)
	y = int(index_y)

	return x, y

######################################################################################################
	
def extractChipDimensions(fileList):
	w = 0
	h = 0
	for file in fileList :
		x, y = getXYframePosition(file)
		
		if (x > w):
			w = x
		if (y > h):
			h = y
		
	print('waffer size: width = %s and height = %s' %(w, h))
	return h, w
###################################################################################################

def fullChipImage(fileList, overlap):
	#creates an image from all pictures of the entire chip
	
	#get chip dimensions
	height, width = extractChipDimensions(fileList)

	tmp = cv2.imread(fileList[0])
	h, w, _ = tmp.shape
	subHeight = int(h * 200 / w)
	subWidth = 200

	# loads all pics from folder and stacks them in a buffer.
	# buffer = []
	# for i in range(0, len(fileList)):
		# tmp = cv2.imread(fileList[i])
		# tmp = cv2.resize(tmp, (subWidth, subHeight))
		# buffer.append(tmp)
	# print('Buffer shape:', np.array(buffer).shape)
	# print('Buffer loaded in %s seconds' %(time.time()- start_time))

	img = np.zeros((int((1-overlap) * subHeight * height + overlap * subHeight), int((1-overlap) * subWidth * width + overlap * subWidth), 3))
	print(img.shape)
	
	idx = 0
	xmin = 0
	xmax = subWidth - 1
	ymin = 0
	ymax = subHeight - 1
	buffer = cv2.imread(fileList[idx])
	buffer = cv2.resize(buffer, (subWidth, subHeight))
	# img[ymin : ymax, xmin : xmax] = buffer[idx][0: subHeight - 1, 0 : subWidth - 1]
	img[ymin : ymax, xmin : xmax] = buffer[0: subHeight - 1, 0 : subWidth - 1]

	for j in range(1,height):
			idx += 1
			ymin = ymax + 1
			ymax = ymin + int((1-overlap) * subHeight) - 1	
			buffer = cv2.imread(fileList[idx])
			buffer = cv2.resize(buffer, (subWidth, subHeight))
			img[ymin : ymax, xmin : xmax] = buffer[int(overlap * subHeight) : subHeight - 1, 0 : subWidth - 1]
			# img[ymin : ymax, xmin : xmax] = buffer[idx][int(overlap * subHeight) : subHeight - 1, 0 : subWidth - 1]

	for i in range(1,width):
		xmin = xmax + 1
		xmax = xmin + int((1-overlap) * subWidth) - 1
		ymin = 0
		ymax = subHeight - 1
		idx += 1
		buffer = cv2.imread(fileList[idx])
		buffer = cv2.resize(buffer, (subWidth, subHeight))
		img[ymin : ymax, xmin : xmax] = buffer[0 : subHeight - 1, int(overlap * subWidth) : subWidth - 1]
		# img[ymin : ymax, xmin : xmax] = buffer[idx][0 : subHeight - 1, int(overlap * subWidth) : subWidth - 1]
		
		for j in range(1,height):
			idx += 1
			ymin = ymax + 1
			ymax = ymin + int((1-overlap) * subHeight) - 1	
			buffer = cv2.imread(fileList[idx])
			buffer = cv2.resize(buffer, (subWidth, subHeight))
			img[ymin : ymax, xmin : xmax] = buffer[int(overlap * subHeight) : subHeight - 1, int(overlap * subWidth) : subWidth - 1]
			# img[ymin : ymax, xmin : xmax] = buffer[idx][int(overlap * subHeight) : subHeight - 1, int(overlap * subWidth) : subWidth - 1]

	return img
#############################################################################################################
def addFlakeToChip(chipImage, overlap, frameNames, flakes):
	
	tmp = cv2.imread(frameNames[0])
	h, w, _ = tmp.shape
	subHeight = int(h * 200 / w)
	subWidth = 200
	
	idx = -1
	for file in frameNames:
		idx += 1
		#recover frame position
		x, y = getXYframePosition(file)
		
		#get top left corner of the frame
		x0 =   (x-1) * int((1-overlap) * subWidth)
		y0 =   (y-1) * int((1-overlap) * subHeight)
		
		for center in flakes[idx]:
			xc = int(center[0]* 200 / w) + x0
			yc = int(center[1]* 200 / w) + y0
			cv2.circle(chipImage, (xc, yc), 10, (0,0,255), 3)		
		
	return chipImage
#############################################################################################################

#try load background image. If doesn't exist, extract one from pictures
if (os.path.isfile(backgroundPath)) :
	backgroundImg = cv2.imread(backgroundPath)
	#if the image and background don't have the same depth, convert the background to the depth of the image.
	if (bitdepth != backgroundImg.dtype) :
		print('WARNING ! Background and images don\'t have the same bitdepth. Precision loss is expected')
		d1 = np.iinfo(backgroundImg.dtype).max
		d2 = np.iinfo(bitdepth).max
		backgroundImg= backgroundImg.astype(np.float32)
		backgroundImg = backgroundImg * d2 / d1
		backgroundImg = backgroundImg.astype(bitdepth)
else :
	print('Extracting background')
	backgroundImg = backgroundCorrection(fileList, maxMemory, bitdepth)
	#save image for future use
	cv2.imwrite(backgroundPath, backgroundImg)

#check if output directory exists. Else create it
if (not os.path.isdir(exportDir)):
	os.makedirs(exportDir)

start_time = time.time()
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
	
	#if there is some significant activity on the frame, analyze it
	if (activityFilter(img, backgroundImg, minArea)):
				
		if (ultra):
			flakesCentroid, img = ultraCourseDetector(img, backgroundImg, minArea, exportImg, distThreshold)
		else :
			flakesCentroid,img = hBNDetector(img, K, HSVClustering, RGBDistance, minThickness, maxThickness, thicknessFilter, minArea, colorTableHSV, thicknessTable, backgroundImg, targetColorHSV, percentage, exportImg, distThreshold)
		
		if ((len(flakesCentroid) > 0)) : 
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

#if required, create a picture of the full chip and add circles where the flakes are
if (fullChip):
	print('Creating full Chip Image')
	start_time = time.time()
	fullImg = fullChipImage(fileList, overlap)
	print('titme to create image : %s seconds' %(time.time() - start_time))
	fullImg = addFlakeToChip(fullImg, overlap, framesWithFlakes, flakes)
	cv2.imwrite(exportDir + '/fullChipWithFlakes.png', fullImg)
print('all Done')