import re
import json
import pandas as pd
import nltk
from tqdm import tqdm
from datasets import load_from_disk
from nltk.tokenize import word_tokenize
from .m2scorer.m2scorer import evaluate

'''
    m2scorer is taken from:
    https://github.com/keisks/m2scorer/tree/master?tab=readme-ov-file
'''

def extract_corrected_text(model_response: str) -> str:
    """
    Extracts the corrected text from the model's response.

    Parameters:
    - model_response (str): The full response from the model.

    Returns:
    - str: The extracted neutralized text, or None if not found.
    """
    ans=model_response.split('\n\n')[0]
    rationale= ' '.join(model_response.split('\n\n')[1:])
    if '\n' in ans:
        ans=ans.split('\n')[1]
                       
    return rationale,ans

def get_model_answer(row):
    # TODO: Currently, only evaluating the first generation for each example.
    _, answer= extract_corrected_text(row['model_output'][0])
    answer=' '.join(word_tokenize(answer))
    return answer

def get_model_rationale(row):
    # TODO: Currently, only evaluating the first generation for each example.
    rationale, _= extract_corrected_text(row['model_output'][0])
    return rationale
    
def get_score(data_path, model_output_path, m2_file_path, reference_col):
    '''
        data_path = Path to HF dataset which has GT
        output_path = path to json file containing model outputs
    '''
    test = pd.DataFrame(load_from_disk(data_path))    
    df = pd.read_json(model_output_path)

    df['gt_answer']= test[reference_col]
    df['model_rationale']= df.apply(get_model_rationale,axis=1)
    df['model_answer']= df.apply(get_model_answer,axis=1)
    
    f1_score, score = evaluate(m2_file_path, df['model_answer'].tolist())
    df['score']=score

    # Write back the exact scores to output_path
    records = df.to_dict(orient = 'records')
    formatted_json = json.dumps(records, indent=4)

    with open(model_output_path,'w') as f:
        f.write(formatted_json)

    return f1_score

def main():
    # data_test = load_from_disk("../datasets/gsm8k/test")
    data_path ="./datasets/neutralization/feedback/"
    model_output_path = f"./outputs/exp-5.0.3.1/feedback/0-shot/eval_1/generated_outputs.json" 
    input_col='input'
    output_col='edits'
    score = get_score(data_path, model_output_path, input_col, output_col)
    print("The score of the model is: ",score)

if __name__=='__main__':
    main()