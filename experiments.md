### exp-0
 - Purpose: Set up few-shot prompting baseline
 - evaluation data : gsm8k
 - model: off-the-shelf instruction tuned models
 - n-shot: 8
 - #### Eval 1
    - Model: LLaMA 3.2 1B
 - #### Eval 2
    - Model: LLaMA 3.2 3B
- #### Eval 3
    - Model: LLaMA 3.1 8B


### exp-1.1
 - Purpose: SFT on off-the-shelf LLaMA 3.2 1B using full model finetuning using zero-shot examples.
 - training data: custom gsm8k train split
 - #### Eval 1
    - Purpose: Inference with 0-shot
 - #### Eval 2
    - Purpose: Inference with 8-shot
    

### exp-1.2
 - Purpose: SFT on off-the-shelf LLaMA 3.2 1B using LoRA finetuning using zero-shot examples.
 - training data: custom gsm8k train split
 - #### Eval 1
    - Purpose: Inference with 0-shot
 - #### Eval 2
    - Purpose: Inference with 8-shot

### exp-1.3
 - Purpose: SFT on off-the-shelf LLaMA 3.2 3B using full model finetuning using zero-shot examples.
 - training data: custom gsm8k train split
 - #### Eval 1
    - Purpose: Inference with 0-shot
 - #### Eval 2
    - Purpose: Inference with 8-shot

### exp-1.4
 - Purpose: SFT on off-the-shelf LLaMA 3.2 3B using LoRA finetuning using zero-shot examples.
 - training data: custom gsm8k train split
 - #### Eval 1
    - Purpose: Inference with 0-shot
 - #### Eval 2
    - Purpose: Inference with 8-shot