import numpy as np
import cv2 as cv

path1 = "dataset_pano/control/Beechwood_0_int/rgb_pano/000_0001.npy"
# path2 = "dataset/control/Beechwood_0_int/rgb/000_0050.npy"

img1 = np.load(path1, allow_pickle=True)
print(img1.shape)
img1 = img1[:,:,2::-1]

# img2 = np.load(path2, allow_pickle=True)
# img2 = img2[:,:,::-1]

cv.imshow("p1", img1)
# cv.imshow("p2", img2)
cv.waitKey(0)
cv.destroyAllWindows()