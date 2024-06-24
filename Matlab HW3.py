#_*_ coding:utf-8_*_
import numpy as np
import cv2

#Step1. 將照片使用cv2的套件引入後再將照片的size改成532x300
img_org = cv2.imread('img.jpg')
img = cv2.resize(img_org,(532,300))

#Step2. 再將照片使用SVD的方法做壓縮，取K=300,200,100,50
def svd_compression(img, k):
    res_image = np.zeros_like(img)
    for i in range(img.shape[2]):
        U, Sigma, VT = np.linalg.svd(img[:,:,i])
        res_image[:, :, i] = U[:,:k].dot(np.diag(Sigma[:k])).dot(VT[:k,:])
 
    return res_image

res1 = svd_compression(img, k=300)
res2 = svd_compression(img, k=200)
res3 = svd_compression(img, k=100)
res4 = svd_compression(img, k=50)

#Step3. 將取不同的K值的圖分別繪製出來
row11 = np.hstack((res1, res2))
row22 = np.hstack((res3, res4))
res = np.vstack((row11, row22))
 
cv2.imshow('img', res)
cv2.waitKey(0)
cv2.destroyAllWindows()


