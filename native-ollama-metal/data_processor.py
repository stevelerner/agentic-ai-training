#!/usr/bin/env python3
"""
Extract character dialogue from Alice in Wonderland for training.
"""

import re
from typing import List, Dict


def extract_mad_hatter_dialogue(text: str) -> List[Dict[str, str]]:
    """Extract all dialogue from the Mad Hatter character."""
    examples = []
    
    # Normalize text
    normalized_text = text.replace('\n', ' ')
    
    # Unicode curly quote characters
    left_quote = '\u201c'  # "
    right_quote = '\u201d'  # "
    
    # Patterns to find dialogue attributed to Hatter
    patterns = [
        f'{re.escape(left_quote)}(.*?){re.escape(right_quote)}\\s+said the Hatter',
        f'{re.escape(left_quote)}(.*?){re.escape(right_quote)},\\s+said the Hatter',
        f'the Hatter[^.]*?said[^.]*?{re.escape(left_quote)}(.*?){re.escape(right_quote)}',
        f'Hatter[^.]*?said[^.]*?,\\s*{re.escape(left_quote)}(.*?){re.escape(right_quote)}',
        f'The Hatter[^.]*?{re.escape(left_quote)}(.*?){re.escape(right_quote)}',
        f'Hatter[^.]*?said[^.]*?was[^.]*?{re.escape(left_quote)}(.*?){re.escape(right_quote)}',
        r'"([^"]+)"\s+said the Hatter',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, normalized_text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            dialogue = match.group(1).strip()
            dialogue = re.sub(r'\s+', ' ', dialogue)
            if len(dialogue) > 5:
                examples.append({
                    "instruction": "Respond as the Mad Hatter from Alice in Wonderland",
                    "input": "",
                    "output": dialogue
                })
    
    # Extract from tea party scene (Chapter VII)
    tea_party_start = text.find("CHAPTER VII")
    tea_party_end = text.find("CHAPTER VIII", tea_party_start)
    
    if tea_party_start != -1 and tea_party_end != -1:
        tea_party_text = text[tea_party_start:tea_party_end]
        tea_party_normalized = tea_party_text.replace('\n', ' ')
        
        hatter_patterns = [
            f'{re.escape(left_quote)}(.*?){re.escape(right_quote)}\\s+said the Hatter',
            f'{re.escape(left_quote)}(.*?){re.escape(right_quote)},\\s+said the Hatter',
            f'the Hatter[^.]*?{re.escape(left_quote)}(.*?){re.escape(right_quote)}',
            f'Hatter[^.]*?said[^.]*?{re.escape(left_quote)}(.*?){re.escape(right_quote)}',
        ]
        
        for pattern in hatter_patterns:
            matches = re.finditer(pattern, tea_party_normalized, re.IGNORECASE | re.DOTALL)
            for match in matches:
                dialogue = match.group(1).strip()
                dialogue = re.sub(r'\s+', ' ', dialogue)
                if len(dialogue) > 5:
                    examples.append({
                        "instruction": "Respond as the Mad Hatter from Alice in Wonderland",
                        "input": "",
                        "output": dialogue
                    })
    
    # Remove duplicates
    seen = set()
    unique_examples = []
    for ex in examples:
        output = ex["output"]
        if output not in seen:
            seen.add(output)
            unique_examples.append(ex)
    
    return unique_examples


def save_training_data(examples: List[Dict[str, str]], output_path: str):
    """Save training examples to JSONL format."""
    import json
    
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Saved {len(examples)} training examples to {output_path}")


if __name__ == "__main__":
    with open('alice_in_wonderland.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Strip Project Gutenberg header/footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    
    examples = extract_mad_hatter_dialogue(text)
    print(f"Extracted {len(examples)} Mad Hatter dialogue examples")
    
    import os
    os.makedirs('training_data', exist_ok=True)
    save_training_data(examples, 'training_data/mad_hatter_training.jsonl')
    
    print("\nSample examples:")
    for i, ex in enumerate(examples[:3]):
        print(f"\n{i+1}. {ex['output'][:100]}...")
