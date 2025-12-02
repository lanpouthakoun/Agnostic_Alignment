from datasets import load_dataset

def convert_to_prompt_format(example):
    """Convert the text field to the HH-RLHF prompt format"""
    text = example["text"]
    
    # Format as conversation with truncated assistant response
    return {
        "prompt": f"\n\nHuman: {text}"
    }

def main():
    dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
    
    # Convert to prompt format
    formatted_dataset = dataset.map(convert_to_prompt_format)
    
    # Keep only the prompt column
    formatted_dataset = formatted_dataset.remove_columns([
        "id", "text", "label", "span_start", "span_end", 
        "span_type", "position_bucket", "source_span_index"
    ])
    
    # Split into train/test (e.g., 80% train, 20% test)
    split_dataset = formatted_dataset.train_test_split(test_size=0.2, seed=42)
    
    print(f"Train size: {len(split_dataset['train'])}")
    print(f"Test size: {len(split_dataset['test'])}")
    print(split_dataset['train'][0])
    
    # Save both splits
    split_dataset['train'].to_parquet("simple_train.parquet")
    split_dataset['test'].to_parquet("simple_test.parquet")

if __name__ == "__main__":
    main()