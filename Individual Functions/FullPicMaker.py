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
parser.add_argument('--overlap', default = .1, type=float, help='Overlap between frames.')
parser.add_argument('--width', required = True, type=int, help='Number of frames along the width')
parser.add_argument('--height', required = True, type=int, help='Number of frames along the height')

args = parser.parse_args()
print(args)

folder = args.folder
width = args.width
height = args.height
overlap = args.overlap
###################################################################################################

start_time = time.time()

#gets the list of all microscope images in given folder
fileList = glob.glob("./" + folder + "/*.png") 


tmp = cv2.imread(fileList[0])
h, w, _ = tmp.shape
subHeight = int(h * 200 / w)
subWidth = 200

#loads all pics from folder and stacks them in a buffer.
buffer = []
for i in range(0, len(fileList)):
	tmp = cv2.imread(fileList[i])
	tmp = cv2.resize(tmp, (subWidth, subHeight))
	buffer.append(tmp)
print('Buffer shape:', np.array(buffer).shape)
print('Buffer loaded in %s seconds' %(time.time()- start_time))


img = np.zeros((int((1-overlap) * subHeight * height + overlap * subHeight), int((1-overlap) * subWidth * width + overlap * subWidth), 3))
print(img.shape)

idx = 0
xmin = 0
xmax = subWidth - 1
ymin = 0
ymax = subHeight - 1
img[ymin : ymax, xmin : xmax] = buffer[idx][0: subHeight - 1, 0 : subWidth - 1]

for j in range(1,height):
		idx += 1
		ymin = ymax + 1
		ymax = ymin + int((1-overlap) * subHeight) - 1	
		img[ymin : ymax, xmin : xmax] = buffer[idx][int(overlap * subHeight) : subHeight - 1, 0 : subWidth - 1]

for i in range(1,width):
	xmin = xmax + 1
	xmax = xmin + int((1-overlap) * subWidth) - 1
	ymin = 0
	ymax = subHeight - 1
	idx += 1
	img[ymin : ymax, xmin : xmax] = buffer[idx][0 : subHeight - 1, int(overlap * subWidth) : subWidth - 1]
	
	for j in range(1,height):
		idx += 1
		ymin = ymax + 1
		ymax = ymin + int((1-overlap) * subHeight) - 1	
		img[ymin : ymax, xmin : xmax] = buffer[idx][int(overlap * subHeight) : subHeight - 1, int(overlap * subWidth) : subWidth - 1]

#save image 
cv2.imwrite('fullPicture.png', img)