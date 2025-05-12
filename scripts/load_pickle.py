import pickle
import os
from pathlib import Path

def load_pickle_data(pickle_path: str):
    """Load data from a pickle file with proper error handling."""
    try:
        # Convert to Path object for better path handling
        pickle_path = Path(pickle_path)
        
        # Check if file exists
        if not pickle_path.exists():
            raise FileNotFoundError(f"Pickle file not found at: {pickle_path}")
            
        # Load the pickle file
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            
        return data
        
    except pickle.UnpicklingError as e:
        print(f"Error unpickling file: {e}")
        raise
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    pickle_path = "data/math/outputs/3b-instruct-out/eval_510/all_outputs.pkl"
    
    try:
        data = load_pickle_data(pickle_path)
        print(f"Successfully loaded pickle data")
        print(f"Data type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys in data: {list(data.keys())}")
        elif isinstance(data, list):
            print(f"Number of items: {len(data)}")
    except Exception as e:
        print(f"Failed to load pickle data: {e}") 