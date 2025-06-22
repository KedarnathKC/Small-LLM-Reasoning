import argparse
import json
from pathlib import Path
from jinja2 import Template, meta
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from api import API
import logging
import hashlib
import traceback
import time
import signal
import sys
import os
# Add Hugging Face datasets import
try:
    from datasets import load_from_disk, load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    logging.warning("Hugging Face datasets not available. Install with: pip install datasets")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GracefulKiller:
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.kill_now = True

class PromptGenerator:
    """Class to handle prompt generation from template files using Jinja2."""
    
    def __init__(self, template_path: str):
        """
        Initialize the prompt generator with a template file.
        
        Args:
            template_path (str): Path to the template file (can be .txt, .md, .prompt, etc.)
        """
        self.template_path = Path(template_path)
        
        # Read the template file
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                self.template_content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Template file not found: {template_path}")
        except Exception as e:
            raise Exception(f"Error reading template file {template_path}: {str(e)}")
        
        # Create Jinja2 template
        try:
            self.template = Template(self.template_content)
        except Exception as e:
            raise Exception(f"Error creating Jinja2 template: {str(e)}")
        
        # Extract variables from the template
        self._extract_variables()
    
    def _extract_variables(self):
        """Extract all variables used in the template."""
        try:
            # Parse the template to find all variables
            from jinja2 import meta, Environment
            env = Environment()
            ast = env.parse(self.template_content)
            self.variables = list(meta.find_undeclared_variables(ast))
        except Exception as e:
            logger.warning(f"Could not extract variables from template: {str(e)}")
            self.variables = []
    
    def generate_prompt(self, **kwargs) -> str:
        """
        Generate a prompt by filling in the template with provided variables.
        
        Args:
            **kwargs: Variables to fill into the template
            
        Returns:
            str: The generated prompt
        """
        try:
            return self.template.render(**kwargs)
        except Exception as e:
            logger.error(f"Error rendering template: {str(e)}")
            raise
    
    def get_required_variables(self) -> List[str]:
        """
        Get the list of required variables for this template.
        
        Returns:
            List[str]: List of variable names
        """
        return self.variables
    
    def get_template_preview(self, max_lines: int = 10) -> str:
        """
        Get a preview of the template content.
        
        Args:
            max_lines: Maximum number of lines to show
            
        Returns:
            str: Preview of the template
        """
        lines = self.template_content.split('\n')
        preview_lines = lines[:max_lines]
        if len(lines) > max_lines:
            preview_lines.append(f"... ({len(lines) - max_lines} more lines)")
        return '\n'.join(preview_lines)


class CheckpointManager:
    """Manages checkpointing for synthetic data generation."""
    
    def __init__(self, output_dir: Path, session_id: Optional[str] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory to save checkpoints
            session_id: Optional session ID for checkpoint naming
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id
        
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.session_id}.json"
        self.samples_file = self.checkpoint_dir / f"samples_{self.session_id}.json"
        self.errors_file = self.checkpoint_dir / f"errors_{self.session_id}.json"
    
    def save_checkpoint(self, 
                       current_var_idx: int, 
                       current_sample_idx: int, 
                       total_samples: List[Dict],
                       failed_samples: List[Dict],
                       config: Dict) -> None:
        """Save checkpoint data."""
        try:
            checkpoint_data = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "current_var_idx": current_var_idx,
                "current_sample_idx": current_sample_idx,
                "total_generated": len(total_samples),
                "total_failed": len(failed_samples),
                "config": config
            }
            
            # Save checkpoint metadata
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Save samples data
            with open(self.samples_file, 'w') as f:
                json.dump(total_samples, f, indent=2)
            
            # Save error data
            with open(self.errors_file, 'w') as f:
                json.dump(failed_samples, f, indent=2)
            
            logger.info(f"Checkpoint saved: {len(total_samples)} samples, {len(failed_samples)} errors at {self.checkpoint_file}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
    
    def load_checkpoint(self) -> Optional[Tuple[int, int, List[Dict], List[Dict], Dict]]:
        """Load checkpoint data if it exists."""
        try:
            if not self.checkpoint_file.exists():
                return None
            
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Load samples
            samples = []
            if self.samples_file.exists():
                with open(self.samples_file, 'r') as f:
                    samples = json.load(f)
            
            # Load errors
            errors = []
            if self.errors_file.exists():
                with open(self.errors_file, 'r') as f:
                    errors = json.load(f)
            
            logger.info(f"Loaded checkpoint: {len(samples)} samples, {len(errors)} errors")
            
            return (
                checkpoint_data["current_var_idx"],
                checkpoint_data["current_sample_idx"],
                samples,
                errors,
                checkpoint_data["config"]
            )
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            return None
    
    def cleanup_checkpoint(self):
        """Clean up checkpoint files after successful completion."""
        try:
            for file in [self.checkpoint_file, self.samples_file, self.errors_file]:
                if file.exists():
                    file.unlink()
            logger.info("Checkpoint files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoint files: {str(e)}")


class ErrorHandler:
    """Handles and categorizes different types of errors during generation."""
    
    ERROR_CATEGORIES = {
        'api_error': 'API request failed',
        'template_error': 'Template rendering failed',
        'validation_error': 'Variable validation failed',
        'timeout_error': 'Request timed out',
        'rate_limit_error': 'Rate limit exceeded',
        'unknown_error': 'Unknown error occurred'
    }
    
    @staticmethod
    def categorize_error(error: Exception) -> str:
        """Categorize error type based on exception."""
        error_str = str(error).lower()
        
        if 'timeout' in error_str or 'timed out' in error_str:
            return 'timeout_error'
        elif 'rate limit' in error_str or 'too many requests' in error_str:
            return 'rate_limit_error'
        elif 'api' in error_str or 'request' in error_str or 'connection' in error_str:
            return 'api_error'
        elif 'template' in error_str or 'jinja' in error_str:
            return 'template_error'
        elif 'validation' in error_str or 'variable' in error_str:
            return 'validation_error'
        else:
            return 'unknown_error'
    
    @staticmethod
    def create_error_record(
        error: Exception,
        variables: Dict,
        prompt: str,
        sample_idx: int,
        var_idx: int,
        config: Dict
    ) -> Dict:
        """Create a detailed error record."""
        error_category = ErrorHandler.categorize_error(error)
        
        return {
            "error_id": hashlib.md5(f"{var_idx}_{sample_idx}_{str(error)}".encode()).hexdigest()[:8],
            "timestamp": datetime.now().isoformat(),
            "error_category": error_category,
            "error_description": ErrorHandler.ERROR_CATEGORIES.get(error_category, "Unknown error"),
            "error_message": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
            "sample_index": sample_idx,
            "variable_set_index": var_idx,
            "input_variables": variables,
            "generated_prompt": prompt,
            "config_used": config,
            "retry_recommended": error_category in ['timeout_error', 'rate_limit_error', 'api_error']
        }


class SyntheticDataGenerator:
    """Class to handle synthetic data generation and output with checkpointing and error handling."""
    
    def __init__(self, 
                 api: API, 
                 prompt_generator: PromptGenerator, 
                 output_dir: str,
                 checkpoint_interval: int = 5,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 session_id: Optional[str] = None):
        """
        Initialize the synthetic data generator.
        
        Args:
            api: API instance for generating completions
            prompt_generator: PromptGenerator instance for creating prompts
            output_dir: Directory to save output files
            checkpoint_interval: Save checkpoint every N variable sets
            max_retries: Maximum number of retries for failed requests
            retry_delay: Base delay between retries (exponential backoff)
            session_id: Optional session ID for checkpointing
        """
        self.api = api
        self.prompt_generator = prompt_generator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_interval = checkpoint_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.output_dir, session_id)
        
        # Initialize graceful shutdown handler
        self.killer = GracefulKiller()
        
        # Track statistics
        self.stats = {
            'total_attempts': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'retries_attempted': 0,
            'checkpoints_saved': 0
        }
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.info(f"Retrying after {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})")
                    time.sleep(delay)
                    self.stats['retries_attempted'] += 1
                
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                error_category = ErrorHandler.categorize_error(e)
                
                # Don't retry certain types of errors
                if error_category in ['template_error', 'validation_error']:
                    break
                
                if attempt < self.max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
        
        raise last_exception
    
    def generate_samples(
        self,
        variables_list: List[Dict[str, str]],
        n_samples: int,
        model: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
        **kwargs
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate synthetic samples for a list of variable sets with checkpointing and error handling.
        
        Args:
            variables_list: List of dictionaries containing variables for each generation
            n_samples: Number of samples to generate per variable set
            model: Model to use for generation
            max_tokens: Maximum tokens per generation
            temperature: Temperature for generation
            **kwargs: Additional parameters for the API
            
        Returns:
            Tuple of (successful_samples, failed_samples)
        """
        # Configuration for checkpointing
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n_samples": n_samples,
            "total_variable_sets": len(variables_list),
            "checkpoint_interval": self.checkpoint_interval,
            "max_retries": self.max_retries,
            **kwargs
        }
        
        # Try to load existing checkpoint
        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        
        if checkpoint_data:
            start_var_idx, start_sample_idx, all_samples, failed_samples, saved_config = checkpoint_data
            
            # Verify config compatibility
            if saved_config.get('model') != model or saved_config.get('n_samples') != n_samples:
                logger.warning("Checkpoint config differs from current config. Starting fresh.")
                start_var_idx, start_sample_idx = 0, 0
                all_samples, failed_samples = [], []
            else:
                logger.info(f"Resuming from checkpoint: var_set {start_var_idx}, sample {start_sample_idx}")
        else:
            start_var_idx, start_sample_idx = 0, 0
            all_samples, failed_samples = [], []
        
        total_generations = len(variables_list) * n_samples
        current_generation = len(all_samples) + len(failed_samples)
        
        try:
            for var_idx in range(start_var_idx, len(variables_list)):
                if self.killer.kill_now:
                    logger.info("Graceful shutdown requested. Saving checkpoint...")
                    break
                
                variables = variables_list[var_idx]
                logger.info(f"Processing variable set {var_idx + 1}/{len(variables_list)}: {variables}")
                
                # Determine starting sample index
                start_sample = start_sample_idx if var_idx == start_var_idx else 0
                
                for sample_idx in range(start_sample, n_samples):
                    if self.killer.kill_now:
                        logger.info("Graceful shutdown requested. Saving checkpoint...")
                        break
                    
                    current_generation += 1
                    self.stats['total_attempts'] += 1
                    
                    logger.info(f"Generating sample {sample_idx + 1}/{n_samples} "
                              f"(Total: {current_generation}/{total_generations})")
                    
                    # Generate prompt
                    try:
                        prompt = self.prompt_generator.generate_prompt(**variables)
                    except Exception as e:
                        logger.error(f"Failed to generate prompt: {str(e)}")
                        error_record = ErrorHandler.create_error_record(
                            e, variables, "", sample_idx, var_idx, config
                        )
                        failed_samples.append(error_record)
                        self.stats['failed_generations'] += 1
                        continue
                    
                    # Get completion with retry logic
                    try:
                        response = self._retry_with_backoff(
                            self.api.get_completion,
                            prompt=prompt,
                            model=model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **kwargs
                        )
                        
                        sample = {
                            "sample_id": hashlib.md5(f"{var_idx}_{sample_idx}_{prompt[:100]}".encode()).hexdigest()[:8],
                            "input_variables": variables,
                            "prompt": prompt,
                            "generated_text": response["text"],
                            "model": response["model"],
                            "provider": response["provider"],
                            "generation_params": {
                                "max_tokens": max_tokens,
                                "temperature": temperature,
                                **kwargs
                            },
                            "timestamp": datetime.now().isoformat(),
                            "sample_index": sample_idx,
                            "variable_set_index": var_idx,
                            "retry_count": getattr(response, 'retry_count', 0)
                        }
                        
                        all_samples.append(sample)
                        self.stats['successful_generations'] += 1
                        
                    except Exception as e:
                        logger.error(f"Error generating sample after all retries: {str(e)}")
                        error_record = ErrorHandler.create_error_record(
                            e, variables, prompt, sample_idx, var_idx, config
                        )
                        failed_samples.append(error_record)
                        self.stats['failed_generations'] += 1
                
                # Save checkpoint after each variable set (or at specified intervals)
                if (var_idx + 1) % self.checkpoint_interval == 0 or var_idx == len(variables_list) - 1:
                    self.checkpoint_manager.save_checkpoint(
                        var_idx + 1, 0, all_samples, failed_samples, config
                    )
                    self.stats['checkpoints_saved'] += 1
                
                if self.killer.kill_now:
                    break
        
        except Exception as e:
            logger.error(f"Critical error during generation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Save emergency checkpoint
            self.checkpoint_manager.save_checkpoint(
                var_idx, sample_idx, all_samples, failed_samples, config
            )
            raise
        
        # Final checkpoint save
        if not self.killer.kill_now:
            self.checkpoint_manager.save_checkpoint(
                len(variables_list), 0, all_samples, failed_samples, config
            )
        
        return all_samples, failed_samples
    
    def save_results(self, 
                    samples: List[Dict], 
                    failed_samples: List[Dict],
                    filename: Optional[str] = None):
        """
        Save generated samples and failed attempts to files.
        
        Args:
            samples: List of successful samples
            failed_samples: List of failed sample attempts
            filename: Optional custom filename (without extension)
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"synthetic_data_{timestamp}"
        
        # Save successful samples
        if samples:
            output_path = self.output_dir / f"{filename}.json"
            with open(output_path, 'w') as f:
                json.dump(samples, f, indent=2)
            logger.info(f"Saved {len(samples)} successful samples to {output_path}")
        
        # Save failed samples
        if failed_samples:
            failed_path = self.output_dir / f"{filename}_failed.json"
            with open(failed_path, 'w') as f:
                json.dump(failed_samples, f, indent=2)
            logger.info(f"Saved {len(failed_samples)} failed attempts to {failed_path}")
        
        # Save detailed summary with statistics
        summary = {
            "generation_summary": {
                "total_samples": len(samples),
                "successful_samples": len(samples),
                "failed_samples": len(failed_samples),
                "success_rate": len(samples) / (len(samples) + len(failed_samples)) * 100 if (len(samples) + len(failed_samples)) > 0 else 0,
                "unique_variable_sets": len(set(str(s["input_variables"]) for s in samples)),
                "generation_timestamp": datetime.now().isoformat(),
                "output_file": str(output_path) if samples else None,
                "failed_file": str(failed_path) if failed_samples else None
            },
            "error_analysis": self._analyze_errors(failed_samples),
            "generation_stats": self.stats,
            "retry_analysis": {
                "total_retries": self.stats['retries_attempted'],
                "avg_retries_per_failure": self.stats['retries_attempted'] / max(1, self.stats['failed_generations'])
            }
        }
        
        summary_path = self.output_dir / f"{filename}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved detailed summary to {summary_path}")
        
        # Clean up checkpoint files on successful completion
        if not self.killer.kill_now:
            self.checkpoint_manager.cleanup_checkpoint()
    
    def _analyze_errors(self, failed_samples: List[Dict]) -> Dict:
        """Analyze error patterns from failed samples."""
        if not failed_samples:
            return {"total_errors": 0}
        
        error_categories = {}
        error_types = {}
        retryable_errors = 0
        
        for error in failed_samples:
            category = error.get('error_category', 'unknown_error')
            error_type = error.get('error_type', 'Unknown')
            
            error_categories[category] = error_categories.get(category, 0) + 1
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            if error.get('retry_recommended', False):
                retryable_errors += 1
        
        return {
            "total_errors": len(failed_samples),
            "error_categories": error_categories,
            "error_types": error_types,
            "retryable_errors": retryable_errors,
            "non_retryable_errors": len(failed_samples) - retryable_errors
        }


def detect_data_source_type(filepath: str) -> str:
    """
    Detect the type of data source.
    
    Returns:
        str: One of 'json_file', 'hf_disk', 'hf_hub'
    """
    path = Path(filepath)
    
    # Check if it's an existing file or directory
    if path.exists():
        if path.is_file() and path.suffix == '.json':
            return 'json_file'
        elif path.is_dir():
            # Check if it's a saved HF dataset (has dataset_info.json or dataset_dict.json)
            if (path / 'dataset_info.json').exists() or (path / 'dataset_dict.json').exists():
                return 'hf_disk'
            else:
                raise ValueError(f"Directory {filepath} exists but doesn't appear to be a saved HF dataset")
    
    # Not a local path - check if it's a HF Hub dataset pattern
    hf_patterns = [
        r'^[a-zA-Z0-9_-]+$',  # Simple dataset name
        r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$',  # user/dataset format
        r'^[a-zA-Z0-9_-]+(/[a-zA-Z0-9_.-]+)?::.+$',  # Contains :: separator
    ]
    
    for pattern in hf_patterns:
        if re.match(pattern, filepath):
            return 'hf_hub'
    
    # If nothing matches, raise error
    raise ValueError(f"Could not determine data source type for: {filepath}")


def parse_hf_config_string(config_string: str) -> Dict[str, Union[str, List[str], int, bool, None]]:
    """
    Parse HuggingFace dataset configuration string.
    
    Format: [dataset_path]::[config_name]::[split]::[columns]::[max_samples]::[streaming]
    """
    parts = config_string.split("::")
    
    config = {
        "dataset": parts[0],
        "config_name": parts[1] if len(parts) > 1 and parts[1] else None,
        "split": parts[2] if len(parts) > 2 and parts[2] else "train",
        "columns": [col.strip() for col in parts[3].split(",")] if len(parts) > 3 and parts[3] else None,
        "max_samples": int(parts[4]) if len(parts) > 4 and parts[4] else None,
        "streaming": parts[5].lower() in ['true', '1', 'yes', 'on'] if len(parts) > 5 else False
    }
    
    return config


def load_hf_dataset_unified(
    source: str,
    source_type: str,
    config: Dict[str, Union[str, List[str], int, bool, None]]
) -> List[Dict[str, str]]:
    """
    Unified loader for HuggingFace datasets from disk or hub.
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("Hugging Face datasets library is not installed. Install with: pip install datasets")
    
    try:
        # Load dataset based on source type
        if source_type == 'hf_disk':
            logger.info(f"Loading HuggingFace dataset from disk: {source}")
            dataset = load_from_disk(source)
            
            # Handle dataset dict (multiple splits)
            if hasattr(dataset, '__getitem__') and config['split'] in dataset:
                dataset = dataset[config['split']]
            elif hasattr(dataset, 'keys') and callable(dataset.keys):
                available_splits = list(dataset.keys())
                if config['split'] not in available_splits:
                    logger.warning(f"Split '{config['split']}' not found. Available: {available_splits}. Using first split.")
                    dataset = dataset[available_splits[0]]
                else:
                    dataset = dataset[config['split']]
        else:  # hf_hub
            logger.info(f"Loading HuggingFace dataset from hub: {config['dataset']}")
            dataset = load_dataset(
                config['dataset'],
                config['config_name'],
                split=config['split'],
                streaming=config['streaming']
            )
        
        # Get available columns
        if hasattr(dataset, 'column_names'):
            available_columns = dataset.column_names
        elif hasattr(dataset, 'features'):
            available_columns = list(dataset.features.keys())
        else:
            available_columns = None
        
        if available_columns:
            logger.info(f"Available columns: {available_columns}")
        
        # Validate requested columns
        if config['columns'] and available_columns:
            invalid_columns = [col for col in config['columns'] if col not in available_columns]
            if invalid_columns:
                raise ValueError(f"Columns {invalid_columns} not found. Available: {available_columns}")
        
        # Convert to list of dictionaries
        variables_list = []
        
        # Handle regular datasets
        dataset_size = len(dataset)
        logger.info(f"Dataset size: {dataset_size}")
        
        if config['max_samples'] and config['max_samples'] < dataset_size:
            logger.info(f"Limiting to {config['max_samples']} samples")
            dataset = dataset.select(range(config['max_samples']))
        
        if config['columns']:
            dataset = dataset.select_columns(config['columns'])
        
        for sample in dataset:
            variables_list.append({k: str(v) for k, v in sample.items()})
        
        logger.info(f"Successfully loaded {len(variables_list)} samples")
        
        if variables_list:
            logger.info(f"Sample variables: {list(variables_list[0].keys())}")
            logger.info(f"First sample preview: {str(variables_list[0])[:200]}...")
        
        return variables_list
        
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")


def load_variables_from_file(filepath: str) -> List[Dict[str, str]]:
    """
    Load variables from a JSON file, HuggingFace dataset from disk, or HuggingFace Hub.
    
    Supported formats:
    - JSON file: "data.json", "/path/to/data.json"
    - HF dataset from disk: "/path/to/saved_dataset", "./my_dataset"
    - HF Hub dataset: "squad", "user/dataset", "squad::train::question,context::1000"
    
    Returns:
        List of variable dictionaries with string values
    """
    try:
        # Detect source type
        source_type = detect_data_source_type(filepath)
        logger.info(f"Detected source type: {source_type}")
        
        if source_type == 'json_file':
            # Load JSON file
            logger.info(f"Loading JSON file: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return [{k: str(v) for k, v in item.items()} for item in data if isinstance(item, dict)]
            elif isinstance(data, dict):
                return [{k: str(v) for k, v in data.items()}]
            else:
                raise ValueError(f"Invalid JSON format. Expected list or dict, got: {type(data)}")
        
        elif source_type in ['hf_disk', 'hf_hub']:
            # Parse configuration if present
            if '::' in filepath:
                config = parse_hf_config_string(filepath)
                source = config['dataset']
            else:
                config = {
                    "dataset": filepath,
                    "config_name": None,
                    "split": "train",
                    "columns": None,
                    "max_samples": None,
                    "streaming": False
                }
                source = filepath
            
            return load_hf_dataset_unified(source, source_type, config)
        
    except Exception as e:
        # Provide helpful error messages
        error_msg = f"Failed to load data from '{filepath}': {str(e)}"
        
        if 'ImportError' in str(type(e)):
            error_msg += "\n\nHint: Install Hugging Face datasets with: pip install datasets"
        elif source_type == 'hf_hub':
            dataset_name = filepath.split('::')[0] if '::' in filepath else filepath
            error_msg += f"\n\nHint: Check if dataset exists at https://huggingface.co/datasets/{dataset_name}"
        
        raise Exception(error_msg)

# The main section needs minimal changes - just update the help text:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data with checkpointing and error handling")
    parser.add_argument("--provider", type=str, choices=["openai", "together", "ollama", "llama"], required=True, help="API provider (openai, together, ollama, llama)")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--prompt-file", type=str, help="Path to prompt template file (.txt, .md, .prompt, etc.)")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save output files")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of samples to generate per variable set")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for nucleus sampling")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    
    # Error handling and checkpointing options
    parser.add_argument("--checkpoint-interval", type=int, default=5, 
                       help="Save checkpoint every N variable sets (default: 5)")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum number of retries for failed requests (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=0.5,
                       help="Base delay between retries in seconds (default: 1.0)")
    parser.add_argument("--session-id", type=str, default=None,
                       help="Custom session ID for checkpointing")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing checkpoint if available")
    
    # Variable input options - UPDATED HELP TEXT
    parser.add_argument("--vars", nargs='*', help="Template variables in key=value format")
    parser.add_argument("--vars-file", type=str, 
                   help="Data source - supports: "
                        "1) JSON file: 'data.json', "
                        "2) HF dataset from disk: './saved_dataset/', "
                        "3) HF Hub dataset: 'squad', 'user/dataset', "
                        "4) With options: 'squad::train::question,context::1000'")
    
    args = parser.parse_args()
    
    try:
        # Initialize API
        api = API(provider=args.provider)
        
        # List models if requested
        if args.list_models:
            logger.info("Available models:")
            models = api.list_models()
            for model in models[:20]:  # Show first 20 models
                logger.info(f"  - {model}")
            if len(models) > 20:
                logger.info(f"  ... and {len(models) - 20} more")
            sys.exit(0)
        
        # Load the prompt generator
        prompt_gen = PromptGenerator(args.prompt_file)
        required_vars = prompt_gen.get_required_variables()
        
        # Show template preview
        logger.info(f"Loaded template from: {args.prompt_file}")
        logger.info(f"Template preview:\n{prompt_gen.get_template_preview()}")
        logger.info(f"Required variables: {', '.join(required_vars) if required_vars else 'None detected'}")
        
        # Prepare variables list - SIMPLIFIED SECTION
        variables_list = []
        
        # Load variables from file or HF dataset if provided
        if args.vars_file:
            try:
                loaded_vars = load_variables_from_file(args.vars_file)
                variables_list.extend(loaded_vars)
                logger.info(f"Loaded {len(loaded_vars)} variable sets from: {args.vars_file}")
            except Exception as e:
                logger.error(f"Failed to load variables: {str(e)}")
                sys.exit(1)
        
        # Add command-line variables if provided
        if args.vars:
            template_vars = {}
            for var in args.vars:
                if '=' in var:
                    key, value = var.split('=', 1)
                    template_vars[key] = value
            if template_vars:
                variables_list.append(template_vars)
                logger.info(f"Added command-line variables: {template_vars}")
        
        # Validate that we have variables
        if not variables_list:
            logger.error("Error: No variables provided. Use --vars or --vars-file")
            logger.info(f"\nRequired variables for this template: {', '.join(required_vars)}")
            logger.info("\nExample usage:")
            logger.info(f"  # JSON file:")
            logger.info(f"  python {parser.prog} --provider {args.provider} --model MODEL --prompt-file TEMPLATE --vars-file variables.json")
            logger.info(f"  # Hugging Face dataset:")
            logger.info(f"  python {parser.prog} --provider {args.provider} --model MODEL --prompt-file TEMPLATE --vars-file squad")
            logger.info(f"  # HF dataset with options:")
            logger.info(f"  python {parser.prog} --provider {args.provider} --model MODEL --prompt-file TEMPLATE --vars-file 'squad::train::question,context::1000'")
            logger.info(f"  # Command line vars:")
            logger.info(f"  python {parser.prog} --provider {args.provider} --model MODEL --prompt-file TEMPLATE --vars", end="")
            for var in required_vars:
                logger.info(f" {var}='value'", end="")
            logger.info()
            sys.exit(1)
        
        if args.demo:
            logger.info("Running in demo mode, will only process 5 variable sets and 2 samples per set")
            args.n_samples = 2
            variables_list = variables_list[:5]

        # Validate all variable sets have required variables
        for idx, var_set in enumerate(variables_list):
            missing_vars = [var for var in required_vars if var not in var_set]
            if missing_vars:
                logger.error(f"Error: Variable set {idx} is missing required variables: {', '.join(missing_vars)}")
                logger.info(f"Required variables: {', '.join(required_vars)}")
                logger.info(f"Provided variables: {', '.join(var_set.keys())}")
                logger.info(f"Sample from variable set {idx}: {str(var_set)[:200]}...")
                sys.exit(1)
        
        # Rest of the code remains exactly the same...
        generator = SyntheticDataGenerator(
            api=api,
            prompt_generator=prompt_gen,
            output_dir=args.output_dir,
            checkpoint_interval=args.checkpoint_interval,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            session_id=args.session_id
        )
        
        logger.info(f"Starting generation: {len(variables_list)} variable sets × {args.n_samples} samples = {len(variables_list) * args.n_samples} total generations")
        logger.info(f"Checkpointing every {args.checkpoint_interval} variable sets")
        logger.info(f"Max retries per request: {args.max_retries}")
        
        samples, failed_samples = generator.generate_samples(
            variables_list=variables_list,
            n_samples=args.n_samples,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        generator.save_results(samples, failed_samples)
        
        total_attempts = len(samples) + len(failed_samples)
        success_rate = (len(samples) / total_attempts * 100) if total_attempts > 0 else 0
        
        logger.info(f"\n{'='*50}")
        logger.info(f"GENERATION COMPLETE")
        logger.info(f"{'='*50}")
        logger.info(f"Total attempts: {total_attempts}")
        logger.info(f"Successful: {len(samples)} ({success_rate:.1f}%)")
        logger.info(f"Failed: {len(failed_samples)}")
        logger.info(f"Retries attempted: {generator.stats['retries_attempted']}")
        logger.info(f"Checkpoints saved: {generator.stats['checkpoints_saved']}")
        logger.info(f"Output saved to: {generator.output_dir}")
        
        if failed_samples:
            logger.info(f"\nError Summary:")
            error_analysis = generator._analyze_errors(failed_samples)
            for category, count in error_analysis.get('error_categories', {}).items():
                logger.info(f"  {category}: {count}")
        
        logger.info(f"{'='*50}")
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)