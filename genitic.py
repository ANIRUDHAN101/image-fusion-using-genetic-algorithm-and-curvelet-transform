import numpy as np
from math import log10, sqrt

"""-------------------------------------------------PSNR calculation-----------------------------------------------

"""


# the functon used to calculate the PSNR value of origanal image and filterd image
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 0
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

#the filter bank function
def filter(img, sol):
 
  # average filter
  if sol[0]==1:
    kernel = np.ones((3,3),np.float32)/9
    img = cv.filter2D(img,-1,kernel)

  # gausian filtering
  if sol[1]==1:
    img = cv.GaussianBlur(img, (3,3), 0,borderType=cv.BORDER_DEFAULT)
   
  # median
  if sol[2]==1:
    img = cv.medianBlur(img,3)
  
  #bilateral
  if sol[3]==1:
    img = cv.bilateralFilter(img,3,11,11)
  #boxfilter
  if sol[4]==1:
    img = cv.boxFilter(img,4,(3,3))
  

  return img

# it is the fitness function for genitic algorithm
def fitness_fun(solution, solution_idx):
  i = filter(img,solution)
  fitness = PSNR(img,i)
  return fitness
