import os
import torch
import numpy as np
from DataPrep import dataPrep
from clustering import kMeans_kNN, KMeans_only
import pickle





loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)