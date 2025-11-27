#!/usr/bin/env python3
from typing import List, Tuple, Optional
from datasets import load_dataset

ds = load_dataset("Anthropic/hh-rlhf")
def is_single_turn(example):
    """Check if conversation is single turn (one human message, one assistant response)"""
    chosen = example['chosen']
    
    # Count occurrences of "Human:" and "Assistant:" to determine turns
    human_count = chosen.count('\n\nHuman:')
    assistant_count = chosen.count('\n\nAssistant:')
    
    # Single turn means exactly one human message and one assistant response
    return human_count == 1 and assistant_count == 1

def keep_chosen_only(example):
    """Keep only the chosen field"""
    return {'chosen': example['chosen']}


def truncate_assistant_response(example):
    """Randomly truncate assistant response at random token position"""
    import random
    
    chosen = example['chosen']
    
    # Find where Assistant: starts
    assistant_idx = chosen.find('\n\nAssistant:')
    if assistant_idx == -1:
        return example
    
    # Get the assistant response part
    assistant_start = assistant_idx + len('\n\nAssistant:')
    assistant_response = chosen[assistant_start:]
    
    # Randomly choose k in [0, len(response)]
    k = random.randint(0, len(assistant_response))
    
    # Truncate at position k
    truncated_response = assistant_response[:k]
    
    # Reconstruct
    example['chosen'] = chosen[:assistant_start] + truncated_response
    return example

def main():
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split='train', streaming = True)

    single_turn_dataset = (dataset
        .filter(is_single_turn)
        .map(truncate_assistant_response)
        .remove_columns(['rejected']))
    
    dataset_for_training = single_turn_dataset.map(lambda x: {
    "prompt": x["chosen"],
    }).remove_columns(['chosen'])

    dataset_for_training.to_parquet("hh_rlhf_single_turn_train.parquet")

    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split='test', streaming = True)

    single_turn_dataset = (dataset
        .filter(is_single_turn)
        .map(truncate_assistant_response)
        .remove_columns(['rejected']))
    
    dataset_for_test = single_turn_dataset.map(lambda x: {
    "prompt": x["chosen"],
    }).remove_columns(['chosen'])

    dataset_for_test.to_parquet("hh_rlhf_single_turn_test.parquet")

    count = 0
    for example in dataset_for_training:
        # Process your example here
        print(example)
        

    print(count)
    


if __name__ == "__main__":
    main()
