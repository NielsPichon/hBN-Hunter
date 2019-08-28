##################################################################################################
# Illumination correction
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
parser.add_argument('--background', required = True, type=str, help='Background image')
parser.add_argument('--image', required = True, type=str, help='image to correct illumination of')

args = parser.parse_args()
print(args)

background = cv2.imread(args.background)
image = cv2.imread(args.image)
###################################################################################################

start_time = time.time()

#get bitdepth and check that both images have the same
backgroundDepth = background.dtype
print('background bit depth :', backgroundDepth)
imageDepth = image.dtype
print('image bit depth :', imageDepth)

#if the image and background don't have the same depth, convert the background to the depth of the image.
if (imageDepth != backgroundDepth) :
	print('WARNING ! Background and image don\'t have the same bitdepth. Precision loss is expected')
	background= background.astype(np.float32)
	d1 = np.iinfo(backgroundDepth).max
	d2 = np.iinfo(imageDepth).max
	background = background * d2 / d1
	background = background.astype(imageDepth)
	cv2.imshow('test',background)

#computing the mean color of the background
meanBackgroundColor = (np.mean(background.astype(np.float32), axis = 0))
meanBackgroundColor = np.mean(meanBackgroundColor, axis = 0)
print('mean background color: ', meanBackgroundColor)

#convert to float to avoid truncating
image = image.astype(np.float32)
background = background.astype(np.float32)

#correct image
correctedImage = np.divide(image, background) * meanBackgroundColor

#cast back to original bit depth
correctedImage = (np.clip(np.around(correctedImage),0,255)).astype(imageDepth)

#saving image
cv2.imwrite('corrected_image.png', correctedImage)

print('Correction completed in %s seconds' %(time.time() - start_time))