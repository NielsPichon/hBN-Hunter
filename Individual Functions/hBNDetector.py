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




###################################################################################################
#Parsing Arguments
parser = argparse.ArgumentParser(description='Finds hBN flakes using color clustering')
parser.add_argument('--n_clusters', default = 16, type=int, help='Number of color clusters to look for. Default = 16')
parser.add_argument('--image', required = True, type=str, help='Microscope image to detect suitable hBN flakes on.')
parser.add_argument('--ClusteringSpace', default = 0, type=int, help='Color Space in which to perform the clustering. 1 for HSV, 0 for RGB. Default = 0 (RGB)')
parser.add_argument('--ThicknessInterpSpace', default = 1, type=int, help='Color Space in which to the thickness computation. 1 for HSV, 0 for RGB. Default = 1 (HSV)')
parser.add_argument('--minThickness', default = 31, type=float, help='Minimum acceptable thickness. Default = 31')
parser.add_argument('--maxThickness', default = 100, type=float, help='Maximum acceptable thickness. Default = 100')
parser.add_argument('--thicknessFilter', default = 3, type=float, help='Thickness threshold (in nm) under which clusters will be aggregated. 0 to turn off. Default = 3')
parser.add_argument('--minArea', default = 1000, type=int, help='Minimum acceptable area (in pixels) for a flake. Default = 1000')

args = parser.parse_args()
print(args)
					
K = args.n_clusters
img = cv2.imread(args.image)
HSVClustering = args.ClusteringSpace
RGBDistance = 1 - args.ThicknessInterpSpace
minThickness = args.minThickness
maxThickness = args.maxThickness
thicknessFilter = args.thicknessFilter
minArea = args.minArea

###################################################################################################

#table of colors and associated thicknesses
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
###################################################################################################

start_time = time.time()
#with rescaling first
height, width, channels = img.shape
resized_image = cv2.resize(img, (500, int(height * 500 / width))) 

#denoise the image for better results (SLOW!!!!!!)
# resized_image = cv2.fastNlMeansDenoisingColored(resized_image)

if (HSVClustering == 1):
	Z = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
Z = resized_image.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,labels,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
print("Kmeans clustering performed in %s seconds" %(time.time() - start_time))

# # print clustered image for debug
# center = np.uint8(center)
# res = center[labels.flatten()]
# res2 = res.reshape((resized_image.shape))
# cv2.imshow('openCV cluster',res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

###################################################################################################

start_time2 = time.time()
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
						
###################################################################################################

		
print("Cluster analysis performed in %s seconds" %(time.time() - start_time2))

# #DEBUG#	
# #compute cluster "center of mass" for debug display
# centerOfMass = np.zeros((K, 2), float)
# height, width = labels.shape
# weight = np.zeros(K, float)
# for i in range(0, height) : 
	# for j in range(0, width) : 
		# centerOfMass[labels[i,j], 0] += i
		# centerOfMass[labels[i,j], 1] += j
		# weight[labels[i,j]] += 1

# centerOfMass[:,0] = np.divide(centerOfMass[:,0], weight)
# centerOfMass[:,1] = np.divide(centerOfMass[:,1], weight)

# # print clustered image for debug
# center = np.uint8(center)
# res = center[labels.flatten()]
# res2 = res.reshape((resized_image.shape))

# for i in range(0,K):
	# font = cv2.FONT_HERSHEY_SIMPLEX
	# cv2.putText(res2, str(int(thickness[i])), (int(centerOfMass[i,1]), int(centerOfMass[i,0])) , font, .5, (0,0,0), 1, cv2.LINE_AA)

# cv2.imshow('thickness',res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #END DEBUG#

#segment suitable layer thickness crystals to find right areas.
start_time3 = time.time()

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
				
		# #debug
		# im = np.uint8(np.dstack([mask]*3) )
		# im = cv2.drawContours(im, contours, contourIdx = -1, color = (255,0,0), thickness = 2)
		# cv2.imshow('contours', im)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()	
		
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if(area > minArea):
				M = cv2.moments(cnt)
				flakesArea.append(area)
				flakesCentroid.append([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])
				flakesContours.append(cnt)
				flakesThickness.append(thickness[i])

print("Flake detection performed in %s seconds" %(time.time() - start_time3))				
print("Total computation performed in %s seconds" %(time.time() - start_time))


###################################################################################################

resized_image = cv2.drawContours(resized_image, flakesContours, contourIdx = -1, color = (255,0,0), thickness = 2)
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(0,len(flakesThickness)):
	cv2.putText(resized_image, str(int(flakesThickness[i])), (flakesCentroid[i][0], flakesCentroid[i][1]) , font, .5, (0,0,0), 1, cv2.LINE_AA)
cv2.imshow('detectedFlakes', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()	