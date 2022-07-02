from oct2py import octave as oc
import cv2 as cv
import os
#defining the GA class
import pygad
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
  global img
  i = filter(img,solution)
  fitness = PSNR(img,i)
  return fitness


if __name__ == "__main__":
    path = os.getcwd()
    image1 = "img1.jpg"
    image2 = "img2.jpg"

    """GENITIC ALGORITHM"""

    ga_instance = pygad.GA(num_generations=10,
                        num_parents_mating=2,
                        fitness_func=fitness_fun,
                        sol_per_pop=4,
                        num_genes=5,
                        gene_space=[0,1],
                        mutation_percent_genes=0.01,
                        mutation_type="random",
                        mutation_by_replacement=True,
                        random_mutation_min_val=0.0,
                        random_mutation_max_val=1.0)

    # read the image from drive
    img = cv.imread(image1)
    ga_instance.run()

    print(f"the image {image1} ")
    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    cv.imwrite('img1f.jpg',filter(img,solution))
    img = cv.imread(image2)
    ga_instance.run()

    print(f"the image {image2} ")
    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    cv.imwrite('img2f.jpg',filter(img,solution))
    oc.addpath(path)
    oc.eval("rgb_fussion")


