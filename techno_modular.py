# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn 
import joblib

def test_model(x):

  #Pre-processing of variables
  
  


  #Loading the model
  model = joblib.load("random_forest.joblib")
  
  ans = model.predict(x.reshape(1,-1))

  if ans == 0:
    return ("The borrower will pay the loan on time")
  else:
     return ("Alert!Alert! This looks like a risky loan")









