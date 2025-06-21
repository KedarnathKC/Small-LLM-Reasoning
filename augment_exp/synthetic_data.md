# Synthetic Data Generator

A robust Python tool for generating synthetic data using AI models with advanced checkpointing, error handling, and retry mechanisms. Perfect for creating large-scale datasets with reliability and fault tolerance.

## 🚀 Features

- **Multi-Provider Support**: OpenAI, Together, Ollama, Llama, and more
- **Template-Based Generation**: Use Jinja2 templates for flexible prompt engineering
- **Fault Tolerance**: Automatic checkpointing and resume capabilities
- **Error Handling**: Comprehensive error categorization and retry logic
- **Graceful Shutdown**: Save progress on interruption (Ctrl+C)
- **Detailed Reporting**: Success rates, error analysis, and statistics
- **Batch Processing**: Process multiple variable sets efficiently

## 📦 Installation & Setup

### 1. Install Dependencies

```bash
pip install jinja2 argparse pathlib datetime typing
# Install your preferred AI provider SDK (openai, together, etc.)
```

### 2. Environment Variables Setup

The generator requires API keys for different providers. Set up the following environment variables:

#### Required Environment Variables

```bash
# For Llama provider
export LLAMA_API_KEY="your-llama-api-key-here"

# For Together provider  
export TOGETHER_API_KEY="your-together-api-key-here"

# For OpenAI provider (if using)
export OPENAI_API_KEY="your-openai-api-key-here"

```

## 🔧 Basic Usage

### Quick Start - Single Generation

Generate one sample with inline variables:

```bash
python generator.py \
  --provider llama \
  --model Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --prompt-file prompts/synthetic_data/math_inspired_prompt.txt \
  --output-dir outputs/math_sample_0619 \
  --vars question="Find all real numbers x such that x^2 + 5x + 6 = 0"
```

### Batch Generation - Multiple Samples

Generate multiple samples from a variables file:

```bash
python generator.py \
  --provider llama \
  --model Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --prompt-file prompts/synthetic_data/math_inspired_prompt.txt \
  --output-dir outputs/math_sample_0619 \
  --vars-file prompts/synthetic_data/sample_math_data.json \
  --n-samples 5
```

## 📋 Command Line Options

### Required Parameters
- `--provider`: API provider (`openai`, `together`, `ollama`, `llama`)
- `--model`: Model name to use
- `--prompt-file`: Path to Jinja2 template file

### Generation Parameters
- `--n-samples`: Number of samples per variable set (default: 1)
- `--max-tokens`: Maximum tokens to generate (default: 300)
- `--temperature`: Sampling temperature (default: 0.7)
- `--output-dir`: Output directory (default: ./output)

### Variable Input
- `--vars`: Inline variables (`key=value key2=value2`)
- `--vars-file`: JSON file with variable sets

### Reliability & Performance
- `--checkpoint-interval`: Save progress every N variable sets (default: 5)
- `--max-retries`: Retry failed requests up to N times (default: 3)
- `--retry-delay`: Base delay between retries in seconds (default: 1.0)
- `--session-id`: Custom session identifier for checkpointing
- `--resume`: Resume from existing checkpoint

### Utility
- `--list-models`: Show available models and exit

## 📝 Template System

Create Jinja2 templates with variables for dynamic prompt generation:

**math_inspired_prompt.txt:**
```
Solve the following mathematical problem step by step:

Problem: {{ question }}
Difficulty: {{ difficulty }}
Context: {{ context }}

Provide a detailed solution with explanations.
```

**Variables JSON file (sample_math_data.json):**
```json
[
  {
    "question": "Find all real numbers x such that x^2 + 5x + 6 = 0",
    "difficulty": "intermediate",
    "context": "quadratic equations"
  },
  {
    "question": "Prove that the sum of angles in a triangle is 180 degrees",
    "difficulty": "advanced",
    "context": "geometry proofs"
  }
]
```

## 🛡️ Error Handling & Reliability

### Automatic Retry Logic

The system automatically retries failed requests with exponential backoff:

```bash
# Configure retry behavior
python generator.py \
  --provider llama \
  --model Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --prompt-file prompts/math_template.txt \
  --vars-file math_problems.json \
  --max-retries 5 \
  --retry-delay 2.0 \
  --n-samples 10
```

### Error Categories

The system categorizes errors for better analysis:
- **api_error**: API request failures
- **template_error**: Jinja2 template rendering issues
- **timeout_error**: Request timeouts
- **rate_limit_error**: API rate limiting
- **validation_error**: Variable validation failures
- **unknown_error**: Uncategorized errors

## 🔄 Checkpointing & Resume

### Automatic Checkpointing

Save progress every N variable sets to prevent data loss:

```bash
# Save checkpoint every 10 variable sets
python generator.py \
  --provider llama \
  --model Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --prompt-file prompts/large_dataset_template.txt \
  --vars-file large_dataset.json \
  --n-samples 20 \
  --checkpoint-interval 10 \
  --session-id large_experiment
```

### Resume from Interruption

If your generation is interrupted, resume from the last checkpoint:

```bash
# Resume using the same session-id
python generator.py \
  --provider llama \
  --model Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --prompt-file prompts/large_dataset_template.txt \
  --vars-file large_dataset.json \
  --n-samples 20 \
  --session-id large_experiment  # Same session ID = auto-resume
```

### Resume from Latest Checkpoint

```bash
# Automatically find and resume from the most recent checkpoint
python generator.py \
  --provider llama \
  --model Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --prompt-file prompts/template.txt \
  --vars-file variables.json \
  --resume
```

## 📊 Real-World Scenarios

### Scenario 1: Large-Scale Dataset Generation

Generate 10,000 samples with robust error handling:

```bash
python generator.py \
  --provider llama \
  --model Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --prompt-file prompts/synthetic_data/comprehensive_math_prompt.txt \
  --vars-file datasets/math_problems_2000_sets.json \
  --n-samples 5 \
  --output-dir outputs/math_large_scale_0619 \
  --checkpoint-interval 25 \
  --max-retries 5 \
  --retry-delay 3.0 \
  --session-id math_large_scale_june19
```

### Scenario 2: High-Reliability Production Run

Maximum reliability with frequent checkpoints:

```bash
python generator.py \
  --provider llama \
  --model Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --prompt-file prompts/synthetic_data/production_template.txt \
  --vars-file production_variables.json \
  --n-samples 10 \
  --output-dir outputs/production_run_0619 \
  --checkpoint-interval 5 \
  --max-retries 8 \
  --retry-delay 5.0 \
  --max-tokens 500 \
  --temperature 0.8 \
  --session-id production_v1
```

### Scenario 3: Rate-Limited API Handling

Optimized for APIs with strict rate limits:

```bash
python generator.py \
  --provider openai \
  --model gpt-4 \
  --prompt-file prompts/synthetic_data/careful_template.txt \
  --vars-file rate_limited_vars.json \
  --n-samples 3 \
  --output-dir outputs/rate_limited_0619 \
  --max-retries 10 \
  --retry-delay 10.0 \
  --checkpoint-interval 2 \
  --session-id rate_limited_experiment
```

### Scenario 4: Quick Prototype Testing

Fast iteration for testing templates:

```bash
python generator.py \
  --provider llama \
  --model Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --prompt-file prompts/test_template.txt \
  --vars question="Test problem" difficulty="easy" \
  --n-samples 2 \
  --output-dir outputs/test_0619 \
  --max-tokens 200 \
  --session-id quick_test
```

### Scenario 5: Resume After Interruption

Your long-running job was interrupted - resume it:

```bash
# Original command that was interrupted:
python generator.py \
  --provider llama \
  --model Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --prompt-file prompts/interrupted_job_template.txt \
  --vars-file interrupted_vars.json \
  --n-samples 15 \
  --session-id interrupted_job_v2

# Resume command (same parameters + session-id):
python generator.py \
  --provider llama \
  --model Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --prompt-file prompts/interrupted_job_template.txt \
  --vars-file interrupted_vars.json \
  --n-samples 15 \
  --session-id interrupted_job_v2  # Will auto-resume!
```

### Scenario 6: Analyze and Retry Failed Samples

After completion, analyze failures and retry them:

```bash
# First, check what failed
cat outputs/math_sample_0619/synthetic_data_20250619_143022_summary.json

# Extract failed samples for retry (manual process or script)
# Then retry with higher retry settings:
python generator.py \
  --provider llama \
  --model Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --prompt-file prompts/synthetic_data/math_inspired_prompt.txt \
  --vars-file failed_samples_variables.json \
  --n-samples 5 \
  --max-retries 10 \
  --retry-delay 5.0 \
  --session-id retry_failed_samples
```

## 📁 Output Structure

The generator creates a comprehensive output structure:

```
outputs/math_sample_0619/
├── synthetic_data_20250619_143022.json           # Successful samples
├── synthetic_data_20250619_143022_failed.json    # Failed attempts with details
├── synthetic_data_20250619_143022_summary.json   # Statistics and analysis
└── checkpoints/                                  # Checkpoint files (auto-cleanup)
    ├── checkpoint_20250619_143022.json
    ├── samples_20250619_143022.json
    └── errors_20250619_143022.json
```

### Sample Output Files

**Successful Sample:**
```json
{
  "sample_id": "a1b2c3d4",
  "input_variables": {
    "question": "Find x where x^2 + 5x + 6 = 0",
    "difficulty": "intermediate"
  },
  "prompt": "Solve the following mathematical problem...",
  "generated_text": "To solve x^2 + 5x + 6 = 0, I'll use factoring...",
  "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
  "provider": "llama",
  "timestamp": "2025-06-19T14:30:22.123456",
  "retry_count": 0
}
```

**Failed Sample:**
```json
{
  "error_id": "e1f2g3h4",
  "error_category": "rate_limit_error",
  "error_message": "Rate limit exceeded. Try again later.",
  "input_variables": {...},
  "generated_prompt": "...",
  "retry_recommended": true,
  "timestamp": "2025-06-19T14:35:10.654321"
}
```

**Summary Statistics:**
```json
{
  "generation_summary": {
    "total_samples": 450,
    "successful_samples": 445,
    "failed_samples": 5,
    "success_rate": 98.89
  },
  "error_analysis": {
    "rate_limit_error": 3,
    "timeout_error": 2
  },
  "generation_stats": {
    "retries_attempted": 12,
    "checkpoints_saved": 18
  }
}
```

## 🎯 Best Practices

### 1. **Use Descriptive Session IDs**
```bash
--session-id "math_dataset_v2_june19"
--session-id "production_stories_batch1"
--session-id "experiment_temperature_07"
```

### 2. **Optimize Checkpoint Intervals**
```bash
# For small datasets (< 100 variable sets)
--checkpoint-interval 10

# For large datasets (> 1000 variable sets)
--checkpoint-interval 50

# For API rate-limited scenarios
--checkpoint-interval 5
```

### 3. **Configure Retries Based on Provider**
```bash
# Stable APIs (OpenAI, Together)
--max-retries 3 --retry-delay 1.0

# Rate-limited APIs
--max-retries 8 --retry-delay 5.0

# Local/Ollama (usually more reliable)
--max-retries 2 --retry-delay 0.5
```

### 4. **Monitor Long-Running Jobs**
```bash
# Use screen or tmux for long-running jobs
screen -S generation_job
python generator.py [your-args]
# Ctrl+A, D to detach
# screen -r generation_job to reattach
```

## 🚨 Troubleshooting

### Common Issues

1. **Template Variables Not Found**
   - Check template file for correct variable names
   - Verify JSON variables file format
   - Use `--vars` to test with single variables

2. **API Connection Errors**
   - Increase `--max-retries` and `--retry-delay`
   - Check API credentials and endpoints
   - Verify model name with `--list-models`

3. **Checkpoint Not Resuming**
   - Ensure same `--session-id` is used
   - Check that model and n-samples match original run
   - Look for checkpoint files in output directory

4. **Out of Memory/Disk Space**
   - Reduce `--n-samples` per run
   - Use smaller `--checkpoint-interval`
   - Monitor disk space in output directory

### Getting Help

```bash
# List available models
python generator.py --provider llama --list-models

# Preview template variables
python generator.py --prompt-file your_template.txt --vars dummy=test

# Test with minimal settings
python generator.py --provider llama --model small-model --prompt-file template.txt --vars test=value --n-samples 1
```

## 📈 Performance Tips

1. **Batch Size Optimization**: Use `--n-samples` between 5-20 for good balance
2. **Checkpoint Frequency**: More frequent checkpoints = slower but safer
3. **Retry Strategy**: Higher retries for unreliable networks
4. **Parallel Processing**: Run multiple sessions with different `--session-id`
5. **Monitor Resources**: Watch CPU, memory, and disk usage during large runs


## 🏆 Summary

This synthetic data generator provides enterprise-grade reliability for AI-powered data generation. Whether you're creating small datasets for prototyping or generating thousands of samples for production, the built-in checkpointing, error handling, and retry mechanisms ensure your work is never lost.

**Key Benefits:**
- ✅ **Fault Tolerant**: Never lose progress due to interruptions
- ✅ **Cost Efficient**: Avoid re-running expensive API calls  
- ✅ **Production Ready**: Comprehensive error handling and reporting
- ✅ **Scalable**: Handle datasets of any size with confidence
- ✅ **Debuggable**: Detailed logs and error analysis for troubleshooting

Happy generating! 🚀