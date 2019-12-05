import cv2

def saliency_map(image, is_image, is_eval):
    if is_image:
        image = cv2.imread(image)
    
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    
    #if is_eval >= 0:
    #    #aliencyMap[200:260, 80:130] = 5*saliencyMap[200:260, 80:130]
    #saliencyMap[30:156, 120:214] = 100*saliencyMap[30:156, 120:214]
    temp = 10*saliencyMap[70:200, 70:200]
    saliencyMap[:, :] = 0*saliencyMap[:, :]
    saliencyMap[70:200, 70:200] = temp

    return saliencyMap/255.0

#sm = saliency_map('../../../data/eval2/video_700_0093.png', 1, 1)

#cv2.rectangle(sm, (120,30), (214,156), 3, 2)

#cv2.imwrite("output.png", sm)
#cv2.imshow("Output", sm)
#cv2.waitKey(0)
