'''
Author: Mohit Mayank
Contact: mohitmayank1@gmail.com
Website: mohitmayank.com

Train blank spacy NER on the house rent agreement training data
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

def train_blank_spacy_ner(TRAIN_DATA, iterations=200):
    """
    Train spacy model on the annotation data
    """
    # Train NER from a blank spacy model
    nlp=spacy.blank("en")

    nlp.add_pipe(nlp.create_pipe('ner'))

    nlp.begin_training()

    # Getting the pipeline component
    ner=nlp.get_pipe("ner")

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
        
    # Disable pipeline components you dont need to change
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # TRAINING THE MODEL
    with nlp.disable_pipes(*unaffected_pipes):

        # Training for 30 iterations
        for iteration in range(iterations):

            # shuufling examples  before every iteration
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                            texts,  # batch of texts
                            annotations,  # batch of annotations
                            drop=0.5,  # dropout - make it harder to memorise data
                            losses=losses,
                        )

            if iteration%10==0:
                print("Losses", losses)
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
    nlp = train_blank_spacy_ner(TRAIN_DATA, iterations=400)

    # Step 3: Test on validation data
    print("Testing on validation data...")
    validation_result = test_on_validation_data(nlp)
    # file_name = 'validation_result_blank_learning'
    # validation_result.to_csv(f"../results/{file_name}.csv", index=False)
    # print("Validation file {file_name} saved to results/* ")
    print(validation_result)

    # Step 4: Save the model
    # print("Saving the model...")
    # save_model(nlp, '../models/blank_learned_ner/')