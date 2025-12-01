#!/usr/bin/env python3
"""
Extract character dialogue from Alice in Wonderland for training.
"""

import re
from typing import List, Dict


def extract_mad_hatter_dialogue(text: str) -> List[Dict[str, str]]:
    """
    Extract all dialogue from the Mad Hatter character.
    Returns list of training examples in instruction format.
    """
    examples = []
    
    # Normalize text - replace newlines with spaces for better matching
    normalized_text = text.replace('\n', ' ')
    
    # Unicode curly quote characters used in the text
    left_quote = '\u201c'  # "
    right_quote = '\u201d'  # "
    
    # Pattern to find dialogue attributed to Hatter
    # Text uses curly quotes: "dialogue" said the Hatter
    # Match: opening curly quote, dialogue content, closing curly quote, then "said the Hatter"
    patterns = [
        # "dialogue" said the Hatter (curly quotes - most common format)
        f'{re.escape(left_quote)}(.*?){re.escape(right_quote)}\\s+said the Hatter',
        # "dialogue," said the Hatter (with comma)
        f'{re.escape(left_quote)}(.*?){re.escape(right_quote)},\\s+said the Hatter',
        # The Hatter said "dialogue"
        f'the Hatter[^.]*?said[^.]*?{re.escape(left_quote)}(.*?){re.escape(right_quote)}',
        # Hatter said, "dialogue"
        f'Hatter[^.]*?said[^.]*?,\\s*{re.escape(left_quote)}(.*?){re.escape(right_quote)}',
        # The Hatter was... "dialogue"
        f'The Hatter[^.]*?{re.escape(left_quote)}(.*?){re.escape(right_quote)}',
        # Hatter... but all he said was "dialogue"
        f'Hatter[^.]*?said[^.]*?was[^.]*?{re.escape(left_quote)}(.*?){re.escape(right_quote)}',
        # Also try straight quotes as fallback
        r'"([^"]+)"\s+said the Hatter',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, normalized_text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            dialogue = match.group(1).strip()
            # Clean up dialogue - remove extra whitespace and newlines
            dialogue = re.sub(r'\s+', ' ', dialogue)
            # Filter out very short or empty dialogue
            if len(dialogue) > 5:
                examples.append({
                    "instruction": "Respond as the Mad Hatter from Alice in Wonderland",
                    "input": "",
                    "output": dialogue
                })
    
    # Also extract from tea party scene (Chapter VII) - lots of Hatter dialogue
    tea_party_start = text.find("CHAPTER VII")
    tea_party_end = text.find("CHAPTER VIII", tea_party_start)
    
    if tea_party_start != -1 and tea_party_end != -1:
        tea_party_text = text[tea_party_start:tea_party_end]
        tea_party_normalized = tea_party_text.replace('\n', ' ')
        
        # Look for Hatter dialogue in tea party with curly quotes
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
    
    # Remove duplicates while preserving order
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
    # Test extraction
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
    
    # Save to training data directory
    import os
    os.makedirs('training_data', exist_ok=True)
    save_training_data(examples, 'training_data/mad_hatter_training.jsonl')
    
    # Print a few examples
    print("\nSample examples:")
    for i, ex in enumerate(examples[:3]):
        print(f"\n{i+1}. {ex['output'][:100]}...")

