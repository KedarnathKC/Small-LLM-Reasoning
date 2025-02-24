from trl import SFTTrainer, SFTConfig
from typing import Union, Optional, Callable
from transformers import PreTrainedTokenizerBase, ProcessorMixin
from datasets import Dataset, IterableDataset
from transformers.image_processing_utils import BaseImageProcessor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from trl.trainer.utils import ConstantLengthDataset
from accelerate import PartialState
from trl.data_utils import is_conversational, maybe_apply_chat_template, maybe_convert_to_chatml, pack_examples

# from trl.trainer import ConstantLengthDataset, DataCollatorForCompletionOnlyLM


class CustomizedSFTTrainer(SFTTrainer):
    # Compared to SFTTrainer: we allow add_special_tokens=False during tokenization.
    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: SFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str
    ) -> Union[Dataset, IterableDataset]:
        # Convert the dataset to an IterableDataset if it is a ConstantLengthDataset
        if isinstance(dataset, ConstantLengthDataset):
            return dataset

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().local_main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                    UserWarning,
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                batched = isinstance(formatting_func(next(iter(dataset))), list)

                def _func(example):
                    return {"text": formatting_func(example)}

                dataset = dataset.map(_func, batched=batched, **map_kwargs)

            # If the dataset is prompt-completion, convert it to language modeling type
            if "prompt" in dataset.column_names and "completion" in dataset.column_names:
                key = "messages" if is_conversational(dataset[0]) else "text"

                def concat_prompt_completion(example):
                    return {key: example["prompt"] + example["completion"]}

                dataset = dataset.map(concat_prompt_completion, remove_columns=["prompt", "completion"])

            # Convert the dataset to ChatML if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
            dataset = dataset.map(
                maybe_convert_to_chatml,
                remove_columns="conversations" if "conversations" in dataset.column_names else None,
                **map_kwargs,
            )

            # Apply the chat template if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": processing_class},
                remove_columns="messages" if "messages" in dataset.column_names else None,  # renamed to "text"
                **map_kwargs,
            )

            # Tokenize the dataset if needed
            if not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                def tokenize(ex):
                    if hasattr(args, 'add_special_tokens'):
                        tokenized = processing_class(ex[args.dataset_text_field], add_special_tokens=args.add_special_tokens)
                    else:
                        tokenized = processing_class(ex[args.dataset_text_field])
                    return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}

                dataset = dataset.map(tokenize, **map_kwargs)

            # Pack or truncate
            if packing:
                if args.max_seq_length is None:
                    raise ValueError("When packing is enabled, `max_seq_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"
                dataset = dataset.select_columns("input_ids")
                dataset = dataset.map(
                    pack_examples, batched=True, fn_kwargs={"seq_length": args.max_seq_length}, **map_kwargs
                )
            elif args.max_seq_length is not None:
                dataset = dataset.map(
                    lambda ex: {key: ex[key][: args.max_seq_length] for key in ["input_ids", "attention_mask"]},
                    **map_kwargs,
                )
            # For Liger kernel, ensure only input_ids is present
            if args.use_liger:
                dataset = dataset.select_columns("input_ids")

        return dataset
