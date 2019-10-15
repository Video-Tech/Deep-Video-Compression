import cv2
import numpy as np

def get_mv(image1, image2):
    frame1 = cv2.imread(image1)
    frame2 = cv2.imread(image2)
    
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    next1 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prvs, next1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = cv2.calcOpticalFlowFarneback(prvs, next1, None, 0.5, 5, 15, 3, 5, 1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    
    horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
    vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
    horz = horz.astype('uint8')
    vert = vert.astype('uint8')
    
    cv2.imwrite('opticalflow_horz.pgm', horz)
    cv2.imwrite('opticalflow_vert.pgm', vert)
    #cv2.imshow('Horizontal Component', horz)
    #cv2.imshow('Vertical Component', vert)

get_mv('d', 'd')
