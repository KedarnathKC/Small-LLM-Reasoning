import re

def fromatting_prompts_func(examples):
    formatted_examples = []
    for i in range(len(examples['question'])):
        answer = format_answer(examples['answer'][i])
        text = f'<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: {examples['question'][i]}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}'
        formatted_examples.append(text)
    return formatted_examples

def format_answer(answer):
        answer = re.sub(r'<<.*?>>', '', answer)
        answer = answer.replace('####', 'The final answer is')
        return answer