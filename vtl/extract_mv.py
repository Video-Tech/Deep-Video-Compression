import cv2
import numpy as np

def get_mv(image1, image2):
    frame1 = cv2.imread('eval1/'+image1)
    frame2 = cv2.imread('eval1/'+image2)
    
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    next1 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prvs, next1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = cv2.calcOpticalFlowFarneback(prvs, next1, None, 0.5, 5, 15, 3, 5, 1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    
    horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
    vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
    horz = horz.astype('uint8')
    vert = vert.astype('uint8')
    
    #cv2.imwrite('eval1_mv/'+image2[:-4]+'_before_flow_x_0001.jpg', horz)
    #cv2.imwrite('eval1_mv/'+image2[:-4]+'_before_flow_y_0001.jpg', vert)
    cv2.imwrite('eval1_mv/'+image2[:-4]+'_after_flow_x_0001.jpg', horz)
    cv2.imwrite('eval1_mv/'+image2[:-4]+'_after_flow_y_0001.jpg', vert)
    #cv2.imshow('Horizontal Component', horz)
    #cv2.imshow('Vertical Component', vert)

get_mv('out_0013.png', 'out_0011.png')
