#!/usr/bin/env python3
"""
Execute notebooks 10-23 and save outputs
"""

import json
import subprocess
import sys
from pathlib import Path

# Working model ID
WORKING_MODEL = 'us.anthropic.claude-sonnet-4-6'

notebooks = [
    'aws_notebooks/10_Recursive_RAG_AWS.ipynb',
    'aws_notebooks/11_Multimodal_RAG_AWS.ipynb',
    'aws_notebooks/12_Agentic_RAG_AWS.ipynb',
    'aws_notebooks/13_Corrective_RAG_AWS.ipynb',
    'aws_notebooks/14_Self_RAG_AWS.ipynb',
    'aws_notebooks/15_Tree_of_Thoughts_RAG_AWS.ipynb',
    'aws_notebooks/16_Chain_of_Thought_RAG_AWS.ipynb',
    'aws_notebooks/17_ReAct_RAG_AWS.ipynb',
    'aws_notebooks/18_Memory_Augmented_RAG_AWS.ipynb',
    'aws_notebooks/19_Ensemble_RAG_AWS.ipynb',
    'aws_notebooks/20_Iterative_RAG_AWS.ipynb',
    'aws_notebooks/21_Few_Shot_RAG_AWS.ipynb',
    'aws_notebooks/22_Hierarchical_RAG_AWS.ipynb',
    'aws_notebooks/23_Parent_Child_RAG_AWS.ipynb',
]

print("=" * 70)
print("BATCH NOTEBOOK EXECUTION")
print("=" * 70)
print(f"\nNotebooks to execute: {len(notebooks)}")
print(f"Working model: {WORKING_MODEL}\n")

def update_model_id(notebook_path):
    """Update model IDs in notebook"""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    old_models = [
        'us.anthropic.claude-opus-4-1-20250805-v1:0',
        'anthropic.claude-opus-4-1-20250805-v1:0',
        'us.anthropic.claude-haiku-4-5-20251001-v1:0',
        'anthropic.claude-3-haiku-20240307-v1:0',
        'anthropic.claude-3-sonnet-20240229-v1:0',
    ]
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for i, line in enumerate(cell['source']):
                for old_model in old_models:
                    if old_model in line:
                        cell['source'][i] = line.replace(old_model, WORKING_MODEL)
                        modified = True
    
    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        return True
    return False

def execute_notebook(notebook_path):
    """Execute notebook using nbconvert"""
    print(f"\n{'='*70}")
    print(f"Executing: {notebook_path}")
    print(f"{'='*70}\n")
    
    try:
        # Execute with timeout
        result = subprocess.run(
            [
                'jupyter', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--inplace',
                '--ExecutePreprocessor.timeout=600',
                '--ExecutePreprocessor.kernel_name=python3',
                notebook_path
            ],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {notebook_path}")
            return True
        else:
            print(f"❌ FAILED: {notebook_path}")
            print(f"Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT: {notebook_path}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {notebook_path}")
        print(f"Exception: {str(e)[:500]}")
        return False

# Main execution
successful = []
failed = []
skipped = []

for i, notebook in enumerate(notebooks, 1):
    print(f"\n[{i}/{len(notebooks)}] Processing {Path(notebook).name}...")
    
    # Check if file exists
    if not Path(notebook).exists():
        print(f"  ⚠️  File not found, skipping")
        skipped.append(notebook)
        continue
    
    # Update model IDs
    print(f"  Updating model IDs...")
    updated = update_model_id(notebook)
    if updated:
        print(f"  ✓ Model IDs updated to {WORKING_MODEL}")
    else:
        print(f"  ℹ️  No model IDs to update")
    
    # Execute
    if execute_notebook(notebook):
        successful.append(notebook)
    else:
        failed.append(notebook)

# Summary
print("\n" + "="*70)
print("EXECUTION SUMMARY")
print("="*70)
print(f"\n✅ Successful: {len(successful)}/{len(notebooks)}")
for nb in successful:
    print(f"  ✓ {Path(nb).name}")

if failed:
    print(f"\n❌ Failed: {len(failed)}/{len(notebooks)}")
    for nb in failed:
        print(f"  ✗ {Path(nb).name}")

if skipped:
    print(f"\n⚠️  Skipped: {len(skipped)}/{len(notebooks)}")
    for nb in skipped:
        print(f"  - {Path(nb).name}")

print(f"\n" + "="*70)
print("DONE")
print("="*70)

sys.exit(0 if not failed else 1)
