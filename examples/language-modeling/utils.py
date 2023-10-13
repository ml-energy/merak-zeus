# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Parts of the code here are adapted from https://github.com/huggingface/transformers/blob/v4.15.0/examples/pytorch/language-modeling/run_clm.py
# Parts of the code here are adapted from https://github.com/huggingface/transformers/blob/v4.15.0/examples/pytorch/language-modeling/run_mlm.py

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer
)
from torch.utils.data import Dataset
import pandas as pd
import math
import numpy as np
import os
import torch.distributed as dist


def load_wikitext(cache_dir, validation_split_percentage):
    raw_datasets = load_dataset('wikitext', 'wikitext-103-raw-v1', cache_dir=cache_dir)
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            'wikitext',
            'wikitext-103-raw-v1',
            split=f'train[:{validation_split_percentage}%]',
            cache_dir=cache_dir
        )
    return raw_datasets

# create dataset
def load_data(data_path, cache_dir, validation_split_percentage):
    data_files = {}
    dataset_args = {}
    if data_path is not None:
        data_files["train"] = data_path

    extension = (
        data_path.split(".")[-1]
    )
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = True
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cache_dir, **dataset_args)


    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{validation_split_percentage}%]",
            cache_dir=cache_dir,
            **dataset_args,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{validation_split_percentage}%:]",
            cache_dir=cache_dir,
            **dataset_args,
        )
    return raw_datasets

def check_cache(cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    files = os.listdir(cache_dir)
    exit_file = False
    for file in files:
        f = str(cache_dir+'/'+file)
        if os.path.isfile(f) and not file.startswith('.'):
            exit_file = True
            break
    return exit_file
            

def create_tokenizer(cache_dir, model_name, config):
    tokenizer_kwargs = {
        "cache_dir": cache_dir,
        "use_fast": True,
        "revision": 'main',
        "use_auth_token": None,
    }
    
    # Only rank 0 to download files
    if dist.get_rank() == 0:
        tokenizer = AutoTokenizer.from_pretrained(model_name, 
            config=config,
            **tokenizer_kwargs)
    dist.barrier()
    if dist.get_rank() != 0:
        tokenizer = AutoTokenizer.from_pretrained(model_name, 
            config=config,
            **tokenizer_kwargs)
    dist.barrier()
    return tokenizer

def preprocessing_datasets(datasets, tokenizer_func, model_name):
    column_names = datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = tokenizer_func.model_max_length
    if max_seq_length > 1024:
        max_seq_length = 1024

    # we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.
    if model_name.startswith("bert"):
        def tokenize_function(examples):
            return tokenizer_func(examples[text_column_name], return_special_tokens_mask=True)
    else:
        def tokenize_function(examples):
            output = tokenizer_func(examples[text_column_name])
            return output

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        load_from_cache_file=True,
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    if model_name.startswith("bert"):
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result
    elif model_name.startswith("gpt"):
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result


    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=1,
        load_from_cache_file=True,
    )

    return lm_datasets['train'], lm_datasets['validation']
