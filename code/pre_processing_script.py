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
from spacy.util import minibatch, compounding

##########
# functions
##########

def clean_text(text_str):
    """
    Clean the annotatated text
    """
    for word in ['[start]', '[partyone]', '[partytwo]', '[rent]', '[end]', '[duration]', '{{', "}}"]:
        text_str = text_str.replace(word, "")
    return text_str

def get_training_example(text):
    """
    Extract annotations from one training text file data
    """
    training_example = []
    offset = 0
    for m in re.compile("{{.*?}}\[.*?\]").finditer(text):
        start = m.start() - offset
        val = re.findall("{{.*?}}", m.group())[0]
        val_type = re.findall("\[.*?\]", m.group())[0]
        offset += 4 + 2 + len(val_type) - 2 
        end = start + len(val) - 4
        training_example.append((start, end, clean_text(val), val_type))
    return training_example, clean_text(text)

def all_get_training_examples():
    """
    run on all training files, 
    extract the annotations examples,
    and only keep the ones which have annotations
    """
    all_training_examples = []
    for file_path in glob("../data/Training_data_text/*"):
        with open(file_path, 'rb') as f:
            text = str(f.read(), "utf-8")
        try:
            example = get_training_example(text)
        except Exception as e:
            print(str(e))
            example = None
        if example is not None:
            all_training_examples.append(example)
    # only pick ones which have annotations
    all_training_examples = [x for x in all_training_examples if len(x[0])>0]
    return all_training_examples

def convert_training_examples_for_spacy(all_training_examples):
    """
    convert to spacy training format
    """
    TRAIN_DATA = []
    for exs in all_training_examples:
        entities = [(ex[0], ex[1], ex[3]) for ex in exs[0]]
        TRAIN_DATA.append((exs[1], {"entities": entities}))
    return TRAIN_DATA

def save_model(nlp, model_path='../models/transfer_learned_ner/'):
    """
    save the model
    """
    output_dir = Path(model_path)
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

def load_model(model_path):
    """
    load and return spacy model
    """
    return spacy.load(model_path)

def test_on_validation_data(nlp):
    """
    Run the model to return NER extraction on validation dataset
    Return csv similar to "ValidationSet.csv"
    """
    validation_result = []
    for file in glob("../data/Validation_Data/*"):
        # read and convert the doc to text
        text = docxpy.process(file)
        # extract entities
        doc = nlp(text)
        result = {ent.label_:ent.text for ent in doc.ents}
        result['file_name'] = file
        # save them to df
        validation_result.append(result)
    # transform the result to dataframe
    validation_result = pd.DataFrame(validation_result)
    return validation_result

