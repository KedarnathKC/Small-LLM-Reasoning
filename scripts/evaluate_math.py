import json
import evaluate
from typing import Dict, List, Optional
import os
from datasets import load_from_disk
import hashlib

def get_problem_from_prompt(prompt: str) -> str:
    return prompt.split("Problem: ")[-1].split("<|eot_id|>")[0].strip()

def get_query_hash(problem: str) -> str:
    return hashlib.sha256(problem.encode()).hexdigest()

def remove_boxed(s: str) -> str:
    """Extract the answer from a boxed expression."""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]
    
    left = "\\boxed{"
    assert s[:len(left)] == left
    assert s[-1] == "}"
    return s[len(left):-1]

def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last boxed expression from a string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]

def process_response(response: str) -> str:
    """Extract the answer from a model's response."""
    # Find the last boxed expression in the response
    boxed = last_boxed_only_string(response)
    if boxed:
        return remove_boxed(boxed)
    
    # If no boxed expression, try to find the last line with "Therefore, the final answer is:"
    if "Therefore, the final answer is:" in response:
        last_line = response.split("Therefore, the final answer is:")[-1].strip()
        # Extract content between $ signs if present
        if "$" in last_line:
            indices = [pos for pos, char in enumerate(last_line) if char == "$"]
            if len(indices) >= 2:
                return last_line[indices[0] + 1:indices[-1]]
        return last_line
    
    return response

def process_solution(solution: str) -> str:
    """Extract the answer from the solution."""
    boxed = last_boxed_only_string(solution)
    if boxed:
        return remove_boxed(boxed)
    return solution

def process_vllm_outputs(output_path, hf_dataset_path):
    hf_dataset = load_from_disk(hf_dataset_path)

    assert os.path.exists(os.path.join(output_path, "all_outputs_processed.json"))

    all_outputs_processed = json.load(open(os.path.join(output_path, "all_outputs_processed.json")))

    outputs_lookup = {}
    for x in all_outputs_processed:
        problem_hash = get_query_hash(get_problem_from_prompt(x['prompt']))
        outputs_lookup[problem_hash] = x

    results = {}
    for data in hf_dataset:
        t = {}
        t.update(data)

        if t['type'] not in results:
            results[t['type']] = []

        problem_hash = get_query_hash(data['problem'])

        t["response"] = outputs_lookup[problem_hash]['outputs'][0]
        t["input_prompt"] = outputs_lookup[problem_hash]['prompt']

        results[t['type']].append(t)

    results_path = os.path.join(output_path, "results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

def evaluate_math_problems(results_file: str) -> Dict:
    """Evaluate math problems using the evaluate library for exact matching."""
    # Load the exact match metric
    exact_match = evaluate.load("exact_match")
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    all_predictions = []
    all_references = []
    detailed_results = {}
    
    for key in results.keys():
        detailed_results[key] = []
        for result in results[key]:
            # Extract answers
            prediction = process_response(result["response"])
            reference = process_solution(result["solution"])
            
            # Store for exact match evaluation
            all_predictions.append(prediction)
            all_references.append(reference)
            
            # Store detailed results
            detailed_results[key].append({
                "problem": result["problem"],
                "prediction": prediction,
                "correct_answer": reference,
                "response": result["response"],
                "solution": result["solution"],
                "exact_match": exact_match.compute(
                    predictions=[prediction],
                    references=[reference]
                )["exact_match"]
            })
            
            
    
    # Calculate exact match score
    exact_match_score = exact_match.compute(
        predictions=all_predictions,
        references=all_references
    )
    
    return {
        "exact_match": exact_match_score["exact_match"],
        "total_problems": len(all_predictions),
        "detailed_results": detailed_results
    }

if __name__ == "__main__":
    # Example usage with your results file
    results_path = '/Users/jaysinha/Workspaces/Small-LLM-Reasoning/data/math/outputs/3b-instruct-out-0-shot-final/eval_511'
    hf_dataset_path = '/Users/jaysinha/Workspaces/Small-LLM-Reasoning/data/math/raw_data/4_shot_math_test_data'
    evals_file = os.path.join(results_path, "evals.json")
    results_file = os.path.join(results_path, "results.json")

    if not os.path.exists(results_file):
        process_vllm_outputs(results_path, hf_dataset_path)
    
    evaluation_results = evaluate_math_problems(results_file)

    with open(evals_file, "w") as f:
        json.dump(evaluation_results, f, indent=4)
    
    # Print summary
    print(f"\nEvaluation Results:")
    print(f"Exact match rate: {evaluation_results['exact_match']:.2%}")
    print(f"Total problems: {evaluation_results['total_problems']}")
    k = list(evaluation_results["detailed_results"].keys())[0]
    # Print detailed results for first few problems
    print("\nDetailed Results (first 3 problems):")
    for i, result in enumerate(evaluation_results['detailed_results'][k][:3]):
        print(f"\nProblem {i+1}:")
        print(f"Problem: {result['problem']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Correct answer: {result['correct_answer']}")
        print("-" * 80)