##################################################################################################
# Background computation
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
import glob
from sys import getsizeof



###################################################################################################
#Parsing Arguments
parser = argparse.ArgumentParser(description='Background extraction from microscope images')
parser.add_argument('--folder', required = True, type=str, help='Folder where microscope images are stored')
parser.add_argument('--memory', default = 10**9, type=int, help='Max memory usage for the buffer in Bytes')

args = parser.parse_args()
print(args)

folder = args.folder
maxMemory = args.memory
###################################################################################################

start_time = time.time()

#gets the list of all microscope images in given folder
fileList = glob.glob("./" + folder + "/*.png") 

#register the bit-depth of the input images to export the background in the right bit depth
bitdepthTestImage = cv2.imread(fileList[0])
bitdepth = bitdepthTestImage.dtype
print('image bitdepth:', bitdepth)

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

#save image 
cv2.imwrite('background.png', background)