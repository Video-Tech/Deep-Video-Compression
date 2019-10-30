import cv2
import numpy as np
import os
import os.path
import glob

dir = "data/train2_1"
output = dir + "_mv1"


def get_mv(image1, image2, after):
    frame1 = cv2.imread(dir+'/'+image1)
    frame2 = cv2.imread(dir+'/'+image2)
    
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    next1 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prvs, next1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #flow = cv2.calcOpticalFlowFarneback(prvs, next1, None, 0.5, 5, 15, 3, 5, 1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    
    horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
    vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
    horz = horz.astype('uint8')
    vert = vert.astype('uint8')

    if after:
	cv2.imwrite(output+'/'+image2[:-4]+'_after_flow_x_0001.jpg', horz)
    	cv2.imwrite(output+'/'+image2[:-4]+'_after_flow_y_0001.jpg', vert)
    else:
	cv2.imwrite(output+'/'+image2[:-4]+'_before_flow_x_0001.jpg', horz)
    	cv2.imwrite(output+'/'+image2[:-4]+'_before_flow_y_0001.jpg', vert)
    #cv2.imshow('Horizontal Component', horz)
    #cv2.imshow('Vertical Component', vert)

def get_names(old_names, idx):
	new_names = old_names
	new_names[-1] = idx
	new_names = "_".join(new_names)
	return new_names + '.png'

def get_filenames(images, idx):
	names = images[:-4].split('/')[-1]
	names = names.split('_')
	size = len(names[-1])

	before_idx = (idx-rem) + 1
	after_idx = before_idx + 12
	before_idx = str(before_idx)
	after_idx = str(after_idx)

	if len(before_idx) != size:
		before_idx = before_idx.zfill(size)
	if len(after_idx) != size:
		after_idx = after_idx.zfill(size)
	return get_names(names, before_idx), get_names(names, after_idx)

#get_mv('out_0013.png', 'out_0011.png', true)

if not os.path.exists(output):
	os.mkdir(output)

for images in glob.iglob(dir + '/*png'):
	img_idx = int(images[:-4].split('_')[-1])
	rem = img_idx%13
	if rem == 0 or (img_idx-1)%13 == 0:
		continue
	before, after = get_filenames(images, img_idx)
	if not os.path.exists(dir+'/'+before) or not os.path.exists(dir+'/'+after):
		continue
	original = images.split('/')[-1]
	get_mv(after, original, False)
	get_mv(before, original, True)
	print("Motion Vector of file %s saved in folder %s" % (dir+'/'+original, output))








