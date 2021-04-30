import cv2
import numpy as np

# 导入前面fix.py 计算的镜头内外参数
u = np.load('parameter/u.npy')
v = np.load('parameter/v.npy')
mtx = np.load('parameter/mtx.npy')
dist = np.load('parameter/dist.npy')

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))

camera = cv2.imread('image/WIN_20201110_21_14_14_Pro.jpg')
image = cv2.resize(camera, (1980, 1080))
h1, w1 = image.shape[:2]
dst1 = cv2.undistort(image, mtx, dist, None, newcameramtx)
mapx, mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w1,h1),5)
dst2 = cv2.remap(image,mapx,mapy,cv2.INTER_LINEAR)
cv2.imwrite('image_cut/1.jpg', dst2)
print('OVER !')
cv2.destroyAllWindows()