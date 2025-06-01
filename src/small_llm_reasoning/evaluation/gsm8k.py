import pandas as pd
from tqdm import tqdm
from evaluate import load
import json
from datasets import load_from_disk
import re

def get_gt_answer(row):
    return row['answer'].split('####')[-1].strip()

def get_generated_answer(row):
    # regex_pattern = r"The final answer is ((-?[$0-9.,]{2,})|(-?[0-9]+))"
    regex_pattern = r"The final answer is -?\$?([0-9,]+(?:\.[0-9]+)?)"

    # TODO: Currently, only evaluating the first generation for each example. 
    match = re.search(regex_pattern, row['output'][0])
    if match:
        # Group -1 corresponds to the entire match
        result = match.group(1).replace(",", "")
    else:
        result = "None"

    return result

def get_generated_rationale(row):
    return row['output'][0].split('The final answer is')[0].strip()

def get_score(data_path, model_output_path):
    '''
        data_path = Path to HF dataset which has GT
        model_output_path = path to json file containing model outputs
    '''
    test = pd.DataFrame(load_from_disk(data_path))    
    df = pd.read_json(model_output_path)
    df['gt_answer'] = test.apply(get_gt_answer, axis=1)
    df['model_rationale'] = df.apply(get_generated_rationale,axis=1)
    df['model_answer'] = df.apply(get_generated_answer,axis=1)
    exact_match_metric = load("exact_match")
    
    # df['score'] = (df['model_answer'] == test['GT_Answer']).astype(int)
    for i in range(df.shape[0]):
        df.loc[i,'score'] = exact_match_metric.compute(predictions = [df.iloc[i]['model_answer']], references = [df.iloc[i]['GT_Answer']])['exact_match']
    
    score = df['score'].sum()/df.shape[0]
    # Write back the exact scores to model_output_path
    records = df.to_dict(orient = 'records')
    formatted_json = json.dumps(records, indent=4)

    with open(model_output_path,'w') as f:
        f.write(formatted_json)
    
    return score

def main():
    data_path ="./datasets/gsm8k/raw/test"
    model_model_output_path = f"/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/Small-LLM-Reasoning/outputs/exp-0/test/8-shot/eval_1/generated_outputs.json" 

    score = get_score(data_path, model_model_output_path)
    print("The score of the model is: ",score)

if __name__=='__main__':
    main()
