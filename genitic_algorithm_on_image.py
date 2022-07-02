#defining the GA class
import pygad
import cv2 as cv
import numpy as np
from math import log10, sqrt

"""-------------------------------------------------PSNR calculation-----------------------------------------------

"""

def initilize(self, path, name, no_gen=200, no_par=2):
    """GENITIC ALGORITHM"""

     name = name
     ga_instance = pygad.GA(num_generations=no_gen,
                          num_parents_mating=no_par,
                          fitness_func= fitness_fun,
                          sol_per_pop=4,
                          num_genes=5,
                          gene_space=[0,1],
                          mutation_percent_genes=0.01,
                          mutation_type="random",
                          mutation_by_replacement=True,
                          random_mutation_min_val=0.0,
                          random_mutation_max_val=1.0)
    # read the image from drive
    path = path + name
     img = cv.imread(path,0)

# the functon used to calculate the PSNR value of origanal image and filterd image
def PSNR(self, original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 0
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

#the filter bank function
def filter(self, img, sol):

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
  i =  filter( img,solution)
  fitness =  PSNR( img,i)
  return fitness

def run(self):
  #RUN THE GA
   ga_instance.run()

def save(self):
  #plote the results
   ga_instance.plot_result()

  # Returning the details of the best solution.
  solution, solution_fitness, solution_idx =  ga_instance.best_solution()
  print(f"the results of the image{ name}")
  print("Parameters of the best solution : {solution}".format(solution=solution))
  print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
  print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
  cv.imwrite( name,  filter( img,solution))
