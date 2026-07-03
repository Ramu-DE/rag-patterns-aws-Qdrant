# Jupyter Notebooks

This directory contains Jupyter notebooks for step-by-step execution of the movie search system.

## Notebooks

### 1. Setup OpenSearch (`01_Setup_OpenSearch.ipynb`)
- Create OpenSearch Serverless collection
- Configure security policies
- Set up authentication
- Save configuration

### 2. Index Movies (`02_Index_Movies.ipynb`)
- Load movie data
- Generate vector embeddings with Bedrock
- Create index with vector field
- Index documents
- Verify and test

## Quick Start

```bash
# Install Jupyter
pip install jupyter

# Launch Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

## Execution Order

1. Run `01_Setup_OpenSearch.ipynb` to create infrastructure
2. Run `02_Index_Movies.ipynb` to load and index data
3. Use the Streamlit UI or Python scripts for search

## Requirements

- AWS credentials configured
- Python 3.8+
- Dependencies from `requirements.txt`
- Bedrock access in your AWS account

## Notes

- Notebooks are independent - can run separately
- All configuration is saved to `../config.json`
- Notebooks include explanations and visualizations
- Can be executed cell-by-cell for learning
