/**
Author: Mohit Mayank
    
Train blank spacy NER on the house rent agreement training and predicts the output on validation set. 
**/

# imports
import os
import docxpy
from glob import glob
import pandas as pd
import spacy 
  
nlp = spacy.load('en_core_web_sm') 