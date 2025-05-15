import re
import signal
from typing import Dict, List, Optional
import pandas as pd
import sympy
from sympy.parsing.latex import parse_latex
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    pass

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
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]

def is_equiv(x1: str, x2: str) -> bool:
    """
    Check if two normalized latex strings are mathematically equivalent
    """
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                logger.debug(f"Had some trouble simplifying when comparing {x1} and {x2}")
    except TimeoutError:
        logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except Exception as e:
        logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False

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
    "square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours",
    "km", "units", "\\ldots", "sue", "points", "feet", "minutes", "digits",
    "cents", "degrees", "cm", "gm", "pounds", "meters", "meals", "edges",
    "students", "childrentickets", "multiples", "\\text{s}", "\\text{.}",
    "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"',
    "\\dots",
]

def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold, is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer

def process_model_output(output: str, target_answer: str) -> Dict[str, int]:
    """
    Process model output and compare with target answer
    """
    last_boxed_string = last_boxed_only_string(output)
    if not last_boxed_string:
        return {"exact_match": 0}
    
    unnormalized_answer = remove_boxed(last_boxed_string)
    answer = normalize_final_answer(unnormalized_answer)
    target = normalize_final_answer(target_answer)

    if answer.strip() == target.strip() or is_equiv(answer, target):
        retval = 1
    else:
        retval = 0

    return {"exact_match": retval}

def evaluate_math_dataset(
    model_outputs: List[str],
    dataset_path: str,
    batch_size: int = 1
) -> Dict[str, float]:
    """
    Evaluate model outputs on MATH dataset
    
    Args:
        model_outputs: List of model generated outputs
        dataset_path: Path to the parquet file containing MATH dataset
        batch_size: Batch size for processing (default: 1)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load dataset
    df = pd.read_parquet(dataset_path)
    
    # Process each example
    exact_matches = []
    
    for i, (output, row) in enumerate(tqdm(zip(model_outputs, df.itertuples()), total=len(df))):
        results = process_model_output(output, row.answer)
        exact_matches.append(results["exact_match"])
        
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1} examples. Current accuracy: {sum(exact_matches) / len(exact_matches):.4f}")
    
    # Calculate final metrics
    accuracy = sum(exact_matches) / len(exact_matches)
    
    return {
        "exact_match": accuracy,
        "total_examples": len(exact_matches),
        "correct_examples": sum(exact_matches)
    }

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate MATH dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to MATH dataset parquet file")
    parser.add_argument("--model_outputs", type=str, required=True, help="Path to file containing model outputs (one per line)")
    
    args = parser.parse_args()
    
    # Load model outputs
    with open(args.model_outputs, "r") as f:
        model_outputs = [line.strip() for line in f]
    
    # Run evaluation
    results = evaluate_math_dataset(model_outputs, args.dataset_path)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Exact Match Accuracy: {results['exact_match']:.4f}")
    print(f"Total Examples: {results['total_examples']}")
    print(f"Correct Examples: {results['correct_examples']}") 