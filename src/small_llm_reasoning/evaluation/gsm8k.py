import pandas as pd
from tqdm import tqdm
from evaluate import load
import json
from datasets import load_from_disk
import re

questions =[
    'There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?',
    'If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?',
    'Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?',
    'Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?',
    'Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?',
    'There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?',
    'Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?',
    'Olivia has $23. She bought five bagels for $3 each. How much money does she have left?'
]
answers =[
    'There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6',
    'There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5',
    'Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39',
    'Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8',
    'Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9',
    'There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29',
    'Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33',
    'Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8'
]
eight_shot_messages = []
for i in range(len(questions)):
    eight_shot_messages.extend([
        {
            'role': 'user',
            'content': f'Given the following problem, reason and give a final answer to the problem.\nProblem: {questions[i]}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n'
        },
        {
            'role': 'assistant',
            'content': answers[i]
        }
    ])


def get_answer_dataset(row):
    return row['answer'].split('####')[-1].strip()
def get_answer_model(row):

    # regex_pattern = r"The final answer is ((-?[$0-9.,]{2,})|(-?[0-9]+))"
    regex_pattern = r"The final answer is (?:-?[$0-9.,]*?(-?[0-9]+(?:\.[0-9]+)?))"

    # TODO: Currently, only evaluating the first generation for each example. 
    match = re.search(regex_pattern, row['output'][0])
    if match:
        # Group -1 corresponds to the entire match
        result = match.group(1)
    else:
        result = "None"

    return result

    # return row['answer'].split('The final answer is')[-1].strip()

def get_score(data_path, output_path):
    '''
        data_path = Path to HF dataset which has GT
        output_path = path to json file containing model outputs
    '''
    test = pd.DataFrame(load_from_disk(data_path))    
    df = pd.read_json(output_path)
    # df.drop(columns=['score','model_answer'],inplace=True)
    df['model_answer'] = df.apply(get_answer_model,axis=1)
    test['GT_Answer'] = test.apply(get_answer_dataset, axis=1)
    # print(f'DF Shape: {df.shape}, Test Shape: {test.shape}')
    exact_match_metric = load("exact_match")
    
    # df['score'] = (df['model_answer'] == test['GT_Answer']).astype(int)
    for i in range(df.shape[0]):
        df.loc[i,'score'] = exact_match_metric.compute(predictions = [df.iloc[i]['model_answer']], references = [test.iloc[i]['GT_Answer']])['exact_match']
    
    # df.drop(columns=['model_answer'], inplace=True)
    score = df['score'].sum()/df.shape[0]
    # Write back the exact scores to output_path
    records = df.to_dict(orient = 'records')
    formatted_json = json.dumps(records, indent=4)

    with open(output_path,'w') as f:
        f.write(formatted_json)
    
    return score

def main():
    # data_test = load_from_disk("../datasets/gsm8k/test")
    data_path ="../datasets/gsm8k/test"
    model_output_path = f"../outputs/gsm8k/LLaMA1B/generated_outputs_full_precision.json" 

    score = get_score(data_path, model_output_path)
    print("The score of the model is: ",score)

if __name__=='__main__':
    main()
