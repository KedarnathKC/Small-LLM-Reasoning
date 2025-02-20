### exp-0
 - Purpose : Set up few-shot prompting baseline
 - evaluation data : gsm8k
 - model : off-the-shelf instruction tuned models
 - n-shot : 8

### exp-1.1
 - Purpose : SFT on off-the-shelf LLaMA 3.2 1B using full model finetuning using zero-shot examples.
 - training data : custom gsm8k train split

### exp-1.2
 - Purpose : SFT on off-the-shelf LLaMA 3.2 1B using LoRA finetuning using zero-shot examples.
 - training data : custom gsm8k train split

### exp-1.3
 - Purpose : SFT on off-the-shelf LLaMA 3.2 3B using full model finetuning using zero-shot examples.
 - training data : custom gsm8k train split

### exp-1.4
 - Purpose : SFT on off-the-shelf LLaMA 3.2 3B using LoRA finetuning using zero-shot examples.
 - training data : custom gsm8k train split