import pandas as pd
import numpy as np
import tree

ROWS = 506



x = pd.read_csv('housing.csv', delim_whitespace=' ', header=None, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12], nrows=round(ROWS * 0.75))
y = pd.read_csv('housing.csv', delim_whitespace=' ', header=None, usecols=[13], nrows=round(ROWS * 0.75))

x_t = pd.read_csv('housing.csv', delim_whitespace=' ', header=None, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12],skiprows=round(ROWS * 0.75))
y_t = pd.read_csv('housing.csv', delim_whitespace=' ', header=None, usecols=[13],skiprows=round(ROWS * 0.75))

