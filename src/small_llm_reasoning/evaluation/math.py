import argparse
import json
import sys
import os
from typing import Dict, List, Optional, Any
import warnings
import pandas as pd
from datasets import load_from_disk

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
            return {"score": 0, "model_answer": model_response, "gt_answer": gt_answer}
        
        model_answer = normalize_final_answer(remove_boxed(last_boxed_string))
        
        # Compare the answers
        if model_answer.strip() == gt_answer.strip() or is_equiv(model_answer, gt_answer):
            result = 1
        else:
            result = 0
        
        return {"score": result, "model_answer": model_answer, "gt_answer": gt_answer}
    except Exception as e:
        print(f"Error: {e}")
        return {"score": 0, "model_answer": model_response, "gt_answer": gt_answer}
    
def get_rationale(model_response): 
    return model_response.split("The final answer is:")[0].strip()

def get_gt_answer(vllm_prompt_input_ids, eval_dataset):
    for i in range(len(eval_dataset)):
        if eval_dataset[i]['input_ids']['prompt_token_ids'] == vllm_prompt_input_ids:
            return eval_dataset[i]['answer']
    raise ValueError(f"Prompt input ids not found in eval dataset")

def get_score(eval_data_path, model_output_path):
    eval_dataset = load_from_disk(eval_data_path)
    df = pd.read_json(model_output_path)
    results = []
    for _, row in df.iterrows():
        t= {}
        row_x = row.to_dict()
        t.update(row_x)
        t['model_rationale'] = get_rationale(row_x['model_output'][0])
        t['answer'] = get_gt_answer(row_x['prompt_input_ids'], eval_dataset)
        result = evaluate_response(t['answer'], row_x['model_output'][0])
        t.update(result)
        results.append(t)
    
    df = pd.DataFrame(results)
    
    # get directory of model_output_path
    dir_path = os.path.dirname(model_output_path)

    # get file name of model_output_path
    file_name = os.path.basename(model_output_path)
    # get file name without extension
    file_name_without_extension = os.path.splitext(file_name)[0]

    # save df to json
    df.to_json(os.path.join(dir_path, f"{file_name_without_extension}_eval.json"), orient='records', indent=4)

    score = df['score'].sum()/df.shape[0]
    return score
    

    
