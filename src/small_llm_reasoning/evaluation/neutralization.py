import os
cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache'
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME']=cache_dir
os.environ['HF_HUB_CACHE']=cache_dir+'/hub'

import re
import json
import pandas as pd
import nltk
from typing import List, Union
from tqdm import tqdm
from evaluate import load
from datasets import load_from_disk

# Taken from:
# https://github.com/facebookresearch/EditEval/blob/main/src/preprocessing.py#L16
def normalize_text(text: Union[str, List[str]]) -> str:
    if isinstance(text, str):
        text = text.split(" ")
    untokenized = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(text)
    return untokenized

def extract_neutralized_text(model_response: str) -> str:
    """
    Extracts the neutralized text from the model's response.

    Parameters:
    - model_response (str): The full response from the model.

    Returns:
    - str: The extracted neutralized text, or None if not found.
    """
    pattern = r"(.*?)(?:The neutralized text is\s*[:\-–—\.]*\s*)(.*)"
    match = re.search(pattern, model_response, re.IGNORECASE | re.DOTALL)
    if not match:
        # If marker not found, return full response as before_text and empty after_text
        return model_response.strip(), ''
    before_text = match.group(1).strip()
    # Remove any leading punctuation from the neutralized part
    neutralized = match.group(2).lstrip(" :-.–—").strip()
    return before_text, neutralized

def get_model_answer(row):
    # TODO: Currently, only evaluating the first generation for each example.
    _, answer= extract_neutralized_text(row['output'][0])
    return answer

def get_model_rationale(row):
    # TODO: Currently, only evaluating the first generation for each example.
    rationale, _= extract_neutralized_text(row['output'][0])
    return rationale


def get_references(row, reference_col):
    # In some tasks, we might have more than one reference edits and in some tasks just one reference edit, which will be just a string.
    if isinstance(row[reference_col], str):
        row[reference_col]=[row[reference_col]]
    references=[normalize_text(reference) for reference in row[reference_col]]
    return references

def get_score(data_path, model_output_path, input_col, reference_col):
    '''
        data_path = Path to HF dataset which has GT
        output_path = path to json file containing model outputs
    '''
    test = pd.DataFrame(load_from_disk(data_path))    
    df = pd.read_json(model_output_path)

    # print(df['model_answer'][0])
    df['gt_answer']= test.apply(lambda row: get_references(row, reference_col),axis=1)
    df['model_rationale']= df.apply(get_model_rationale,axis=1)
    df['model_answer']= df.apply(get_model_answer,axis=1)
    
    test[input_col]=test.apply(lambda row: normalize_text(row[input_col]), axis=1)

    sari = load("sari")
    sari_scores=[]
    for i in range(df.shape[0]):
        sari_scores.append(sari.compute(sources=[test[input_col][i]], predictions=[df['model_answer'][i]], references=[df['GT_Answer'][i]])['sari'])
    df['score']=sari_scores
    score = df['score'].sum()/df.shape[0]

    # Write back the exact scores to output_path
    records = df.to_dict(orient = 'records')
    formatted_json = json.dumps(records, indent=4)

    with open(model_output_path,'w') as f:
        f.write(formatted_json)
    return score

def main():
    # data_test = load_from_disk("../datasets/gsm8k/test")
    data_path ="./datasets/neutralization/tokenized/LLaMA3B-Instruct/val/0-shot/"
    model_output_path = f"./outputs/exp-6.2.1.1/val/0-shot/eval_1/generated_outputs.json" 
    input_col='input'
    output_col='edits'
    score = get_score(data_path, model_output_path, input_col, output_col)
    print("The score of the model is: ",score)

if __name__=='__main__':
    main()