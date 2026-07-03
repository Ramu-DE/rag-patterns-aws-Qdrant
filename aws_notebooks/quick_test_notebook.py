#!/usr/bin/env python3
"""Quick test - Execute simplest notebook (22_Hierarchical_RAG) as proof"""

import json
import subprocess
import sys
from pathlib import Path

# Test with Hierarchical RAG - it's self-contained and fast
TEST_NOTEBOOK = 'aws_notebooks/22_Hierarchical_RAG_AWS.ipynb'
WORKING_MODEL = 'us.anthropic.claude-sonnet-4-6'

print("=" * 70)
print("QUICK NOTEBOOK EXECUTION TEST")
print("=" * 70)
print(f"\nTest notebook: {TEST_NOTEBOOK}")
print(f"Working model: {WORKING_MODEL}\n")

# Update model IDs
print("Step 1: Updating model IDs...")
with open(TEST_NOTEBOOK, 'r') as f:
    nb = json.load(f)

old_models = [
    'us.anthropic.claude-opus-4-1-20250805-v1:0',
    'anthropic.claude-opus-4-1-20250805-v1:0',
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
    with open(TEST_NOTEBOOK, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"✓ Updated to {WORKING_MODEL}\n")

# Execute
print("Step 2: Executing notebook...")
print("(This may take 2-5 minutes with AWS API calls)\n")

try:
    result = subprocess.run(
        [
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--inplace',
            '--ExecutePreprocessor.timeout=600',
            TEST_NOTEBOOK
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd='/workshop/rag_pattern_notebooks'
    )
    
    if result.returncode == 0:
        print(f"\n✅ SUCCESS!")
        print(f"\nNotebook executed and saved with outputs")
        print(f"Location: {TEST_NOTEBOOK}")
        print(f"\nYou can now:")
        print(f"  1. Open it in Jupyter to see outputs")
        print(f"  2. Commit and push to GitHub")
        print(f"  3. Use same approach for other notebooks")
        sys.exit(0)
    else:
        print(f"\n❌ EXECUTION FAILED")
        print(f"\nStderr output:")
        print(result.stderr[:1000])
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    sys.exit(1)
