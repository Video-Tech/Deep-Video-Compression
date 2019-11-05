import cv2

def saliency_map(image, is_image):
    if is_image:
        image = cv2.imread(image)
    
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    
    return saliencyMap

#sm = saliency_map('../../../data/eval_new/silent_cif_0096.png', 1)
#cv2.imshow("Output", sm)
#cv2.waitKey(0)
