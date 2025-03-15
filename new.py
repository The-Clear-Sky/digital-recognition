import numpy as np

mask = np.zeros((5,5),dtype=np.uint8)
a = mask[1:3,2:5]
print(mask)
print('-'*30)
print(a)

import cv2
img = cv2.imread('template.png')
img = img[3:40,5:70]
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
