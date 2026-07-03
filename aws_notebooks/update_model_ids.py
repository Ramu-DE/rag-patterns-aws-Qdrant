#!/usr/bin/env python3
"""Update model IDs in notebooks to use working models"""

import json
import glob

# Working model ID
OLD_MODEL = 'us.anthropic.claude-opus-4-1-20250805-v1:0'
NEW_MODEL = 'us.anthropic.claude-sonnet-4-6'

notebooks = glob.glob('aws_notebooks/1*.ipynb') + glob.glob('aws_notebooks/2*.ipynb')
notebooks.sort()

updated_count = 0

for notebook_path in notebooks:
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    modified = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for i, line in enumerate(cell['source']):
                if OLD_MODEL in line:
                    cell['source'][i] = line.replace(OLD_MODEL, NEW_MODEL)
                    modified = True
    
    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        
        updated_count += 1
        print(f"✓ Updated: {notebook_path}")

print(f"\n✅ Updated {updated_count} notebooks")
print(f"   Changed: {OLD_MODEL}")
print(f"   To: {NEW_MODEL}")
