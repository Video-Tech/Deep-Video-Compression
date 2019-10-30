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

#sm = saliency_map('images/neymar.jpg', 1)
