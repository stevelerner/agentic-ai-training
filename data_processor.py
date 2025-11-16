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
    
    # Pattern to find dialogue attributed to Hatter
    # Matches: "dialogue" said the Hatter / Hatter said / etc.
    patterns = [
        r'"([^"]+)"\s+said the Hatter[^.]*\.',
        r'the Hatter[^.]*said[^.]*"([^"]+)"',
        r'Hatter[^.]*"([^"]+)"',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            dialogue = match.group(1).strip()
            if len(dialogue) > 10:  # Filter very short snippets
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
        
        # Extract dialogue lines that are clearly Hatter's
        lines = tea_party_text.split('\n')
        for i, line in enumerate(lines):
            if 'Hatter' in line and '"' in line:
                # Try to extract the quoted dialogue
                quotes = re.findall(r'"([^"]+)"', line)
                for quote in quotes:
                    if len(quote) > 10 and '?' in quote or '!' in quote or len(quote) > 30:
                        examples.append({
                            "instruction": "Respond as the Mad Hatter from Alice in Wonderland",
                            "input": "",
                            "output": quote
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

