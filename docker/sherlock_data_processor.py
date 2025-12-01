#!/usr/bin/env python3
"""
Generate synthetic training data for the "Sherlock Holmes" agent.
This script creates procedural mysteries where the agent must use tools
to find clues and deduce the culprit.
"""

import json
import random
from typing import List, Dict, Any

def generate_synthetic_dataset() -> List[Dict[str, Any]]:
    """Generate synthetic examples of Sherlock Holmes solving mysteries."""
    examples = []
    
    # We will generate 50 unique cases, each duplicated/varied slightly
    for _ in range(50):
        case = generate_mystery_case()
        conversation = build_conversation_from_case(case)
        examples.append({"messages": conversation})
        
    # Duplicate to ensure enough volume
    final_dataset = []
    for _ in range(4):
        final_dataset.extend(examples)
        
    random.shuffle(final_dataset)
    return final_dataset

def generate_mystery_case() -> Dict[str, Any]:
    """Create a consistent mystery state."""
    
    suspects = ["The Gardener", "The Butler", "The Maid", "The Cook", "The Duke"]
    locations = ["Garden", "Kitchen", "Library", "Ballroom", "Cellar"]
    crimes = ["stole the ruby", "poisoned the tea", "broke the vase", "hid the will"]
    
    # 1. Setup the Crime
    culprit = random.choice(suspects)
    crime = random.choice(crimes)
    location = random.choice(locations)
    
    # 2. Create Clues
    # The culprit leaves a clue at the location
    clue_map = {
        "The Gardener": "muddy footprints size 10",
        "The Butler": "a white glove stained with wine",
        "The Maid": "a stray hair ribbon",
        "The Cook": "traces of flour",
        "The Duke": "a signet ring"
    }
    clue = clue_map[culprit]
    
    # 3. Create Statements (Truths and Lies)
    statements = {}
    for s in suspects:
        if s == culprit:
            # Culprit lies or is evasive
            statements[s] = random.choice([
                "I was nowhere near the scene!",
                "I saw nothing, I swear!",
                "I was busy working all day."
            ])
        else:
            # Innocents might give helpful info
            statements[s] = f"I saw {culprit} heading towards the {location} earlier."

    return {
        "culprit": culprit,
        "crime": crime,
        "location": location,
        "clue": clue,
        "statements": statements,
        "suspects": suspects
    }

def build_conversation_from_case(case: Dict[str, Any]) -> List[Dict[str, str]]:
    """Simulate a perfect run of Sherlock solving the case."""
    
    culprit = case["culprit"]
    location = case["location"]
    clue = case["clue"]
    
    # Intro
    intro_q = f"Sherlock, someone {case['crime']}! Who did it?"
    
    conversation = [{"role": "user", "content": intro_q}]
    
    # Step 1: Inspect the scene
    conversation.append({
        "role": "assistant", 
        "content": f"The game is afoot! I must examine the scene immediately. {{\"tool\": \"inspect_scene\", \"arguments\": {{\"location\": \"{location}\"}}}}"
    })
    conversation.append({
        "role": "user", 
        "content": f"Tool output: Found {clue}."
    })
    
    # Step 2: Deduction / Interview (simplified for training)
    # Holmes realizes the clue points to the culprit
    deduction = ""
    if "muddy" in clue: deduction = "Mud! The Gardener!"
    elif "glove" in clue: deduction = "A glove! The Butler!"
    elif "ribbon" in clue: deduction = "A ribbon! The Maid!"
    elif "flour" in clue: deduction = "Flour! The Cook!"
    elif "ring" in clue: deduction = "A ring! The Duke!"
    
    # Optional: Interview the suspect to confirm (50% chance to add this step)
    if random.random() > 0.5:
        conversation.append({
            "role": "assistant",
            "content": f"Interesting. {deduction} I shall interrogate them. {{\"tool\": \"interview_suspect\", \"arguments\": {{\"name\": \"{culprit}\"}}}}"
        })
        stmt = case["statements"][culprit]
        conversation.append({
            "role": "user",
            "content": f"Tool output: {stmt}"
        })
        final_thought = f"They deny it, of course. But the {clue} does not lie."
    else:
        final_thought = f"The evidence is incontrovertible."

    # Final Answer
    conversation.append({
        "role": "assistant",
        "content": f"{final_thought} It was {culprit}! Elementary, my dear Watson!"
    })
    
    return conversation

def format_for_llama3(examples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Format examples for Llama 3 instruction tuning."""
    formatted_data = []
    
    system_prompt = """You are Sherlock Holmes, the world's greatest consulting detective.
You use deductive reasoning and forensic science to solve crimes.
You are cold, logical, and observant.
IMPORTANT: You have access to tools. USE THEM to gather evidence before making a conclusion.
Tools:
- inspect_scene(location): Look for clues.
- interview_suspect(name): Question a person.
"""

    for ex in examples:
        messages = ex['messages']
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            text += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            
        formatted_data.append({"text": text})
        
    return formatted_data

if __name__ == "__main__":
    import os
    
    # Ensure training_data directory exists
    os.makedirs("training_data", exist_ok=True)
    
    print("Generating synthetic Sherlock Holmes mysteries...")
    dataset = generate_synthetic_dataset()
    formatted = format_for_llama3(dataset)
    
    # Split into train/val
    split_idx = int(len(formatted) * 0.9)
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]
    
    print(f"Writing {len(train_data)} training examples...")
    with open("training_data/train.jsonl", "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Writing {len(val_data)} validation examples...")
    with open("training_data/valid.jsonl", "w") as f:
        for entry in val_data:
            f.write(json.dumps(entry) + "\n")
            
    print("Done!")
