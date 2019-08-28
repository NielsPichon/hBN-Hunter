##################################################################################################
# ColorRangeCompression
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
import time
import argparse


###################################################################################################
#Parsing Arguments
parser = argparse.ArgumentParser(description='Background illumination correction')
parser.add_argument('--image', required = True, type=str, help='image to correct illumination of')
parser.add_argument('--stretch', type=str, default = "gray", help='stretching method')
parser.add_argument('--gray', type=float, default = 0.8, help='value of gray (in percentage of the compressed range), if gray strech method is used')

args = parser.parse_args()
print(args)

image = cv2.imread(args.image)
stretchType = args.stretch
grayPercentage = args.gray
###################################################################################################
start_time = time.time()

#get bitdepth
imagedepth = image.dtype
print('image bit depth :', imagedepth)

#convert to grayscale
grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#compute histogram
hist = cv2.calcHist([grayImg], [0], None, [256], [0, 256])

#Compute median value of histogram
medianHist = np.median(hist, axis = 0)
print('median hist value: ', medianHist)

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
print('right most peak : %s ' %(rigthPeakPos))

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
print('left most peak : %s ' %(leftPeakPos))

#compute left range bound
leftBound = np.floor((rigthPeakPos+leftPeakPos)/2)
print('Compressed range : %s to %s' %(leftBound, rigthPeakPos))

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
	print('Unknown stretching type')
	exit()
	
#cast back to original bit depth
correctedImage = (np.clip(np.around(image),0,255)).astype(imagedepth)

#export image
cv2.imwrite('compressed_' + stretchType + '.png', correctedImage)

print('Dynamic range compression completed in %s seconds' %(time.time() - start_time))