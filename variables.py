#Main file containing overall variables
import os

device = 'cpu' #device where models whill be run
DATASET = 'FD001' #which data set to use from cmpass [FD001, FD002, FD003, FD004]

BATCHSIZE = 100
EPOCHS = 100

k = 10 #amount of folds for cross validation