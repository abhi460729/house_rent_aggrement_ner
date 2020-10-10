## House rent agreement extraction

>Including the introduction to the project and steps taken to solve this problem.
>
>---This project is done by Mohit Mayank (www.mohitmayank.com)---

<u>**Problem**</u>: Extract relevant entities from several house rent agreement data (validation data) using NN model trained on training data.

<u>**Solution**</u>:

- I formulated this problem as a "Named entity extraction" (NER) problem.
- Have used Spacy - a python package to solve this problem.
- Spacy provides two ways to train NER
  - Train existing NER, which we called Transfer learning
  - Train new NER, which we called Blank learning
- I have coded and trained both models, but get better result for 2nd i.e. Blank learning
- Run the inference code (below) to test the trained model. Testing code takes time t run, hence packaged the model as well.

**<u>Important files</u>**:

- Inference code: @ code/inference_trained_best_model.py
- Training code: @ code/spacy_blank_ner.py and code/pre_processing_script.py
- Final result: @ results/validation_result_blank_learning.csv
- Final model: @ models/blank_learned_ner/

**<u>Detailed Steps:</u>**

1. Preprocessing
   1. Convert Docx files to Txt files
   2. Manually mark different annotations in the text file -- ***Note I have only marked ~15 out of ~40***. This subset was sufficient to showcase the prowess of this method. ***The results will improve with more training data.***
   3. Read the annotated text and extract the entities marked
   4. Create the training data in the format required by Spacy
2. Model training
   1. Train model 1: on already trained spacy model (comes ready) i.e. transfer learning method
   2. [Best] Train model 2: train a blank NER from scratch.
3. Model test
   1. Run the trained model on  validation data and store the final result and save the model

**<u>Disclaimer:</u>**

1. As the project was just to do extraction, I have focused more on model training than on post processing to clean the extraction
2. There were overlap of data in training data and validation data. I have only trained on data which were missing in validation set
3. There was inconsistency between the training data and the original docs files - I think this was issue of OCR, sharing some examples --
   1. agreement end date was missing from many files
   2. agreement start date was not in clean date format
   3. party name was partial or sometime incorrect
