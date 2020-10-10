'''
Author: Mohit Mayank
Contact: mohitmayank1@gmail.com
Website: mohitmayank.com

Train trained spacy NER on the house rent agreement training data
    and predict the output on validation set. 
'''
##########
# imports
##########
import os
import re
import spacy 
import random
import docxpy
import pandas as pd
from glob import glob
from pathlib import Path
from pre_processing_script import all_get_training_examples, \
    convert_training_examples_for_spacy, test_on_validation_data, save_model
from spacy.util import minibatch, compounding

##########
# variables
##########

# the new labels we want to extract from the data
new_labels = ['[start]', '[partyone]', '[partytwo]', '[rent]', '[end]', '[duration]']

##########
# functions
##########

def train_transfer_learning_spacy_ner(TRAIN_DATA, iterations=200):
    """
    Train spacy model on the annotation data
    """
    # Load pre-existing spacy model
    nlp=spacy.load('en_core_web_sm')

    # Getting the pipeline component
    ner=nlp.get_pipe("ner")

    # Add the new label to ner
    for label in new_labels:
        ner.add_label(label)

    # Resume training
    optimizer = nlp.resume_training()

    # List of pipes you want to train
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    # List of pipes which should remain unaffected in training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # Begin training by disabling other pipeline components
    with nlp.disable_pipes(*other_pipes) :
        sizes = compounding(1.0, 4.0, 1.001)
        # Training for 30 iterations     
        for itn in range(iterations):
            # shuffle examples before training
            random.shuffle(TRAIN_DATA)
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=sizes)
            # ictionary to store losses
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                # Calling update() over the iteration
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            # status
            if itn%10==0:
                print(f"Iteration: {itn}; Losses: {losses}")
    # return
    return nlp

##########
# Main run
##########
if __name__ == "__main__":
    
    # Step 1: Prepare the training data
    print("Laoading the training data...")
    all_training_examples = all_get_training_examples()
    TRAIN_DATA = convert_training_examples_for_spacy(all_training_examples)
    print("\tLoaded {len(TRAIN_DATA)} training data.")

    # Step 2: Run the spacy NER training
    print("Training SPACY NER....")
    nlp = train_transfer_learning_spacy_ner(TRAIN_DATA, iterations=400)

    # Step 3: Test on validation data
    print("Testing on validation data...")
    validation_result = test_on_validation_data(nlp)
    # file_name = 'valiation_result_transfer_learning'
    # validation_result.to_csv(f"../results/{file_name}.csv", index=False)
    # print("Validation file {file_name} saved to results/* ")
    print(validation_result)

    # Step 4: Save the model
    # print("Saving the model...")
    # save_model(nlp)