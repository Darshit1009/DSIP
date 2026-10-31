# immport libraries 
import numpy as np
import cv2
import  matplotlib.pyplot as plt
image = cv2.imread("C:/Users/hp/OneDrive/Pictures/Screenshots/Screenshot 2026-07-26 164051.png", 1)
prewitt_x=np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=np.float32)
prewitt_y=np.array([[-1,-1,-1],[0,0,0],[1,1,1]],dtype=np.float32)
px=cv2.filter2D(image.astype(np.float32),cv2.CV_32F,prewitt_x)
py=cv2.filter2D(image.astype(np.float32),cv2.CV_32F,prewitt_y)
prewitt=cv2.magnitude(px,py)

prewitt=cv2.normalize(prewitt,None,0,255,cv2.NORM_MINMAX).astype(np.unit32)
plt.imshow(prewitt,cmap='gray')
plt.title('Prewitt Operator')
plt.axis('off')
plt.show()
'''roberts_x=np.asarray([[1,0],[0,-1]],dtype=np.float32)
roberts_y=np.asarray([[1,0],[0,-1]],dtype=np.float32)
gx=cv2.filter2D(image.astype(np.float32),cv2.CV_32F,roberts_x)
gy=cv2.filter2D(image.astype(np.float32),cv2.CV_32F,roberts_y)
roberts=cv2.magnitude(gx,gy)
roberts=cv2.normalize(roberts,None,0,255,cv2.NORM_MINMAX).astype(np.uint32)
plt.imshow(roberts,cmap='gray')
plt.title('ROberts Edge detection')
plt.axis('off')
plt.show()'''
'''laplacian=cv2.Laplacian(image,cv2.CV_64F)
print(laplacian)
laplacian_abs=cv2.convertScaleAbs(laplacian)
print(laplacian_abs)
plt.imshow(laplacian_abs,cmap='gray')
plt.title('Laplacian Edge Detection')
plt.axis('off')
plt.show()'''
'''g0=cv2.GaussianBlur(image,(5,5),sigmaX=0)
g1=cv2.GaussianBlur(image,(5,5),sigmaX=1)
g2=cv2.GaussianBlur(image,(5,5),sigmaX=2)
plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(g0,cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv2.cvtColor(g1,cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv2.cvtColor(g2,cv2.COLOR_BGR2RGB))
plt.show()
plt.title('Gaussian Filter')
plt.axis('off')'''

'''weighted_kernel=np.asarray([[1,1,1],[1,2,1],[1,1,1],[1,1,1]],dtype=np.float32)
weighted_kernel=weighted_kernel/13
weighted_mean=cv2.filter2D(image,-1,weighted_kernel)
plt.figure(figsize=(8,6))
plt.imshow(cv2.cvtColor(weighted_mean,cv2.COLOR_BGR2RGB))
plt.title('Weighted Mean Filter')
plt.axis('off')
plt.show()'''
