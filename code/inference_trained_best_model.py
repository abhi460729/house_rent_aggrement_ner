'''
Author: Mohit Mayank
Contact: mohitmayank1@gmail.com
Website: mohitmayank.com

Load the best trained NER model i.e. Blank model
Run on the validation data and return results 
'''

##########
# imports
##########
from pre_processing_script import load_model, test_on_validation_data

##########
# Main run
##########
if __name__ == "__main__":

    # Step 1: Load the best model
    print("Loading the best trained model ie. Blank model")
    nlp = load_model("../models/blank_learned_ner")

    # Step 2: RUn on the validation data
    print("Testing on validation data...")
    validation_result = test_on_validation_data(nlp)
    print(validation_result)

    file_name = 'validation_result_blank_learning'
    validation_result.to_csv(f"../results/{file_name}.csv", index=False)
    print("Validation file {file_name} saved to results/* ")