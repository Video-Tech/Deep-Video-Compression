import os
import cv2

num_layers = 3

images = []
temp1 = cv2.imread('input_images/silent_cif_0001.png')
for _ in range(num_layers - 1):
    images.append(temp1)
    temp2 = temp1.copy()
    temp1 = cv2.GaussianBlur(temp2, (5,5), cv2.BORDER_DEFAULT)
    
    images.append(temp2 - temp1)

images.append(temp1)

for index, image in enumerate(images):
    cv2.imwrite('output/image'+str(index)+'.png', image)