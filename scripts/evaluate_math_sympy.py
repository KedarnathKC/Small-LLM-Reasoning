import argparse
import json
import sys
import os
from typing import Dict, List, Optional, Any
import warnings

from datasets import load_from_disk
import hashlib

def get_problem_from_prompt(prompt: str) -> str:
    return prompt.split("Problem: ")[-1].split("<|eot_id|>")[0].strip()

def get_query_hash(problem: str) -> str:
    return hashlib.sha256(problem.encode()).hexdigest()

# Suppress ANTLR version mismatch warnings
warnings.filterwarnings("ignore", message="ANTLR runtime and generated code versions disagree")

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    import antlr4
    # Try to get version, but don't fail if we can't
    try:
        antlr_version = antlr4.__version__
    except AttributeError:
        # If we can't get version, assume it's the conda version which should be compatible
        antlr_version = "4.11.1"
except ModuleNotFoundError as e:
    print(f"Error: Required packages not found. {e}")
    print("Please install via: conda install -c conda-forge sympy antlr4-python3-runtime=4.11.1")
    sys.exit(1)

# Import utility functions
import re
import signal

class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last boxed expression from a LaTeX string."""
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
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s: str) -> str:
    """Remove the boxed wrapper from a LaTeX string."""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]

def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    Copied from Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer

def is_equiv(x1: str, x2: str) -> bool:
    """
    Check if two normalized latex strings are equivalent using sympy.
    """
    try:
        with timeout(seconds=5):
            try:
                # Clean up LaTeX expressions before parsing
                x1 = x1.replace('\\iny', '\\infty')  # Fix common typo
                x2 = x2.replace('\\iny', '\\infty')
                
                # Handle special cases for intervals
                if ('\\le' in x1 or '\\le' in x2) and ('[' in x1 or '[' in x2):
                    x1 = x1.replace('\\le', '\\left')
                    x2 = x2.replace('\\le', '\\left')
                
                # For simple numeric comparisons, try direct comparison first
                try:
                    if x1.strip('-').replace('.', '').isdigit() and x2.strip('-').replace('.', '').isdigit():
                        return float(x1) == float(x2)
                except ValueError:
                    pass
                
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ) as e:
                if "couldn't parse" not in str(e):  # Only print non-parsing errors
                    print(f"WARNING: {e}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                return False
    except TimeoutError:
        return False
    except Exception as e:
        if "ANTLR" not in str(e):  # Don't print ANTLR version warnings
            print(f"WARNING: Failed comparing {x1} and {x2} with {e}")
        return False

def evaluate_response(ground_truth: str, model_response: str) -> Dict[str, int]:
    """
    Evaluate a model response against a ground truth solution.
    
    Args:
        ground_truth: The ground truth solution from the MATH dataset
        model_response: The response generated by the model
    
    Returns:
        A dictionary with the evaluation results
    """
    # Process the ground truth
    gt_answer = normalize_final_answer(remove_boxed(last_boxed_only_string(ground_truth)))
    try:
        # Process the model response
        last_boxed_string = last_boxed_only_string(model_response)
        if not last_boxed_string:
            print("No boxed string found in model response. Marking as incorrect.")
            return {"exact_match": 0}
        
        model_answer = normalize_final_answer(remove_boxed(last_boxed_string))
        
        # Compare the answers
        if model_answer.strip() == gt_answer.strip() or is_equiv(model_answer, gt_answer):
            result = 1
        else:
            result = 0
        
        return {"exact_match": result}
    except Exception as e:
        print(f"Error: {e}")
        return {"exact_match": 0}

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

def evaluate_problems(results_dir: str, hf_dataset_path: str) -> Dict[str, Any]:
    """
    Evaluate all problems from a results JSON file in the given directory.
    
    Args:
        results_dir: Path to directory containing results.json
        
    Returns:
        Dictionary containing evaluation results and statistics for each category and overall
    """
    results_file = os.path.join(results_dir, "results.json")
    if not os.path.exists(results_file):
        print(f"results.json not found in directory {results_dir}, processing vllm outputs")
        process_vllm_outputs(results_dir, hf_dataset_path)
        
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading results file: {e}")
        sys.exit(1)
    
    # Initialize results structure
    category_results = {}
    total_problems = 0
    total_correct = 0
    all_detailed_results = []
    
    # Process each category
    for category, problems in data.items():
        category_total = len(problems)
        category_correct = 0
        category_detailed = []
        
        for problem in problems:
            ground_truth = problem.get('solution', '')
            model_response = problem.get('response', '')
            
            if not ground_truth or not model_response:
                print(f"Warning: Missing ground truth or model response for a problem in {category}")
                continue
                
            result = evaluate_response(ground_truth, model_response)
            category_correct += result['exact_match']
            
            category_detailed.append({
                'problem': problem.get('problem', ''),
                'prediction': model_response,
                'correct_answer': ground_truth,
                'is_correct': bool(result['exact_match']),
                'level': problem.get('level', ''),
                'type': problem.get('type', '')
            })
        
        # Calculate category metrics
        category_exact_match = category_correct / category_total if category_total > 0 else 0
        category_results[category] = {
            'exact_match': category_exact_match,
            'total_problems': category_total,
            'correct_count': category_correct,
            'detailed_results': category_detailed
        }
        
        # Update overall metrics
        total_problems += category_total
        total_correct += category_correct
        all_detailed_results.extend(category_detailed)
    
    # Calculate overall metrics
    overall_exact_match = total_correct / total_problems if total_problems > 0 else 0
    
    return {
        'overall': {
            'exact_match': overall_exact_match,
            'total_problems': total_problems,
            'correct_count': total_correct,
            'detailed_results': all_detailed_results
        },
        'categories': category_results
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate model responses against ground truth for MATH dataset")
    parser.add_argument("--results_dir", type=str, required=True, 
                      help="Path to directory containing results.json")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--hf_dataset_path", type=str, required=False, default='/scratch3/workspace/wenlongzhao_umass_edu-metakd/dev_jay/small-reasoning-lm/data/math/raw_data/4_shot_math_test_data',
                      help="Path to the Hugging Face dataset")
    
    args = parser.parse_args()
    
    # Evaluate all problems
    evaluation_results = evaluate_problems(args.results_dir, args.hf_dataset_path)
    
    # Save results to evals.json in the same directory
    evals_file = os.path.join(args.results_dir, "evals.json")
    try:
        with open(evals_file, "w") as f:
            json.dump(evaluation_results, f, indent=2)
    except Exception as e:
        print(f"Error writing evals.json: {e}")
        sys.exit(1)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    print("Overall Performance:")
    print(f"Exact match rate: {evaluation_results['overall']['exact_match']:.2%}")
    print(f"Total problems: {evaluation_results['overall']['total_problems']}")
    print(f"Correct answers: {evaluation_results['overall']['correct_count']}")
    print("\nPerformance by Category:")
    print("-" * 50)
    
    # Sort categories by exact match rate for better readability
    sorted_categories = sorted(
        evaluation_results['categories'].items(),
        key=lambda x: x[1]['exact_match'],
        reverse=True
    )
    
    for category, results in sorted_categories:
        print(f"\n{category}:")
        print(f"Exact match rate: {results['exact_match']:.2%}")
        print(f"Problems: {results['total_problems']}")
        print(f"Correct answers: {results['correct_count']}")
    
    if args.verbose:
        print("\nDetailed Results (first 3 problems from each category):")
        print("-" * 80)
        for category, results in sorted_categories:
            print(f"\n{category} (Sample Problems):")
            for i, result in enumerate(results['detailed_results'][:3]):
                print(f"\nProblem {i+1}:")
                print(f"Level: {result['level']}")
                print(f"Problem: {result['problem']}")
                print(f"Prediction: {result['prediction']}")
                print(f"Correct answer: {result['correct_answer']}")
                print(f"Correct: {result['is_correct']}")
                print("-" * 80)
    
    print(f"\nResults saved to: {evals_file}")

if __name__ == "__main__":
    main()