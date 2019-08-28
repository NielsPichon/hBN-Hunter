##################################################################################################
# Flake Detection refinement
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
parser.add_argument('--filter', default = 10, type=float, help='Thickness threshold (in nm) under which clusters will be aggregated. 0 to turn off. Default = 3')
parser.add_argument('--minArea', default = 1000, type=int, help='Minimum acceptable area (in pixels) for a flake. Default = 1000')
parser.add_argument('--noiseFilter', default = 0, type =int, help ='strength of the noise filtering')
parser.add_argument('--kernTolerance', default = 0.05, type =float, help ='tolerance (in percentage) of the device fitting')
parser.add_argument('--minX', default = 40, type =int, help ='min width size of device to fit')
parser.add_argument('--minY', default = 80, type =int, help ='min length size of device to fit')

args = parser.parse_args()
print(args)
					
K = args.n_clusters
img = cv2.imread(args.image)
HSVClustering = args.ClusteringSpace
HSVFilter = args.filter
minArea = args.minArea
noiseFilter = args.noiseFilter
kernelTolerance = args.kernTolerance
min_x = args.minX
min_y = args.minY
###################################################################################################
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
start_time = time.time()


#with rescaling first
height, width, channels = img.shape
resized_image = cv2.resize(img, (500, int(height * 500 / width))) 

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
print("Kmeans clustering performed in %s seconds" %(time.time() - start_time))

# #print clustered image for debug
# center = np.uint8(center)
# res = center[labels.flatten()]
# res2 = res.reshape((resized_image.shape))
# cv2.imshow('openCV cluster',res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# exit()

###################################################################################################

start_time2 = time.time()
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
# cv2.imshow('openCV cluster',res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# exit()
						
###################################################################################################	
print("Cluster analysis performed in %s seconds" %(time.time() - start_time2))

#segment suitable layer thickness crystals to find right areas.
start_time3 = time.time()

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


print("Flake detection performed in %s seconds" %(time.time() - start_time3))				
print("Total computation performed in %s seconds" %(time.time() - start_time))

###################################################################################################
# #print contours
# resized_image = cv2.drawContours(resized_image, flakesContours, contourIdx = -1, color = (255,0,0), thickness = 2)
# font = cv2.FONT_HERSHEY_SIMPLEX
# for i in range(0,len(flakesThickness)):
	# cv2.putText(resized_image, str(int(flakesThickness[i])), (flakesCentroid[i][0], flakesCentroid[i][1]) , font, .5, (0,0,0), 1, cv2.LINE_AA)
# cv2.imshow('detectedFlakes', resized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()	


##################################################################################################
#filter with given kernel to find if such patch exists in the picture
start_time = time.time()
print('Fitting devices ...')

kernelSize = min_x * min_y
objective = kernelSize * (1 - kernelTolerance)

#create kernel of min device size and shape (rectangle only)
kernel = np.ones((min_x, min_y),dtype=np.uint8)

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
				device.append([maxY, maxX])
				deviceRot.append(i)			
				
				# #debug
				# cv2.imshow('mask', mask * 255)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()	
				# im = (np.clip(np.around(255.0 * np.float32(filterIm)/np.float(objective)),0,255)).astype(np.uint8)
				# im[filterIm < max] = 100
				# im = cv2.circle(im, (maxY, maxX), 10, (0,0,255))
				# cv2.imshow('filteredIm', im)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()					
				break
				
print('Device fitting performed in %s seconds' %(time.time() -start_time))
#####################################################################################################
# #print fitted devices
idx = -1
for dev in device:
	idx += 1
	res2 = drawRotateRectange(image = res2, center = dev, angle = deviceRot[idx], width = min_x, height = min_y)
	
cv2.imshow('fitted devices', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()	