# Notebook Execution Guide

## ✅ Setup Complete

All prerequisites installed and tested:
- Python 3.12 virtual environment
- boto3, opensearch-py, jupyter, and dependencies
- AWS credentials configured
- Bedrock API access verified
- Working models identified

## 🚀 How to Execute Notebooks

### Option 1: Execute All Notebooks (Batch)

```bash
cd /workshop/rag_pattern_notebooks
source venv/bin/activate

# Execute notebooks 10-23
for i in {10..23}; do
  notebook="aws_notebooks/$(ls aws_notebooks/ | grep "^${i}_")"
  echo "Executing: $notebook"
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 "$notebook"
done
```

### Option 2: Execute Individual Notebook

```bash
cd /workshop/rag_pattern_notebooks
source venv/bin/activate

# Example: Execute Hierarchical RAG
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  aws_notebooks/22_Hierarchical_RAG_AWS.ipynb
```

### Option 3: Interactive Execution (Recommended)

```bash
cd /workshop/rag_pattern_notebooks
source venv/bin/activate
jupyter notebook

# Then:
# 1. Navigate to aws_notebooks/
# 2. Open any notebook (10-23)
# 3. Click "Run All" or run cells individually
# 4. Save notebook (outputs saved automatically)
```

## 📋 Notebooks 10-23

| # | Pattern | Complexity | Est. Time |
|---|---------|------------|-----------|
| 10 | Recursive RAG | Medium | 5-8 min |
| 11 | Multimodal RAG | High | 8-10 min |
| 12 | Agentic RAG | Medium | 5-7 min |
| 13 | Corrective RAG | Medium | 5-8 min |
| 14 | Self RAG | Medium | 6-9 min |
| 15 | Tree of Thoughts | High | 7-10 min |
| 16 | Chain of Thought | Medium | 5-7 min |
| 17 | ReAct RAG | Medium | 5-8 min |
| 18 | Memory Augmented | High | 8-12 min |
| 19 | Ensemble RAG | High | 7-10 min |
| 20 | Iterative RAG | Medium | 5-7 min |
| 21 | Few-Shot RAG | Low | 3-5 min |
| 22 | Hierarchical RAG | Low | 3-5 min |
| 23 | Parent-Child RAG | Medium | 5-7 min |

**Total estimated time**: 1-2 hours for all 14 notebooks

## ⚙️ Configuration

All notebooks now use working models:
```python
EMBEDDING_MODEL = 'amazon.titan-embed-text-v2:0'  # ✅ Working
LLM_MODEL = 'us.anthropic.claude-sonnet-4-6'       # ✅ Working
AWS_REGION = 'us-west-2'                           # ✅ Configured
```

## 💰 Cost Estimate

**Per notebook**: ~$0.10-0.50 depending on complexity
**All 14 notebooks**: ~$3-7 total

Breakdown:
- Embeddings: ~$0.01 per notebook
- LLM calls (Sonnet 4.6): ~$0.09-0.49 per notebook
- OpenSearch: $0.24/hour (collection already running)

## 🔧 Troubleshooting

### Error: "No module named 'aws_utils'"
```bash
# Make sure you're in the right directory
cd /workshop/rag_pattern_notebooks
source venv/bin/activate
```

### Error: "OpenSearch 404"
```bash
# OpenSearch indexes are created automatically by notebooks
# Just ensure config.json exists with correct endpoint
cat config.json
```

### Error: "Model access denied"
```bash
# Use working model IDs (already updated in notebooks):
# us.anthropic.claude-sonnet-4-6  ✅
# NOT: anthropic.claude-3-haiku-20240307-v1:0  ❌
```

### Notebook hangs/times out
```bash
# Increase timeout (default 600s):
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 \
  your_notebook.ipynb
```

## 📤 Committing Outputs to GitHub

After executing notebooks:

```bash
cd /workshop/rag_pattern_notebooks

# Add all notebooks with outputs
git add aws_notebooks/1*.ipynb aws_notebooks/2*.ipynb

# Commit
git commit -m "Add execution outputs for notebooks 10-23

Executed with:
- Model: us.anthropic.claude-sonnet-4-6
- Embeddings: amazon.titan-embed-text-v2:0
- Region: us-west-2
- OpenSearch: qrm9kbjh7wmnpa99ee2b.us-west-2.aoss.amazonaws.com

All outputs saved in notebook cells

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Push to GitHub
git push origin main
```

## 🎯 Quick Test (Single Notebook)

Test the setup with the fastest notebook:

```bash
cd /workshop/rag_pattern_notebooks
source venv/bin/activate

# Execute Few-Shot RAG (fastest, ~3 min)
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  aws_notebooks/21_Few_Shot_RAG_AWS.ipynb

# Check outputs
jupyter notebook aws_notebooks/21_Few_Shot_RAG_AWS.ipynb
```

## 📊 Execution Status Tracking

Create a simple tracker:

```bash
echo "Notebook,Status,Time,Date" > execution_log.csv

# After each notebook:
echo "10_Recursive_RAG_AWS.ipynb,SUCCESS,5.2min,$(date)" >> execution_log.csv
```

## 🔐 Security Reminder

**IMPORTANT**: After this session, rotate the AWS credentials that were shared:
```bash
aws sts get-session-token  # Get new temporary credentials
```

Never commit credentials to Git!

## ✅ What's Ready

- [x] Python environment with all packages
- [x] AWS credentials configured
- [x] Bedrock access verified  
- [x] Titan embeddings working (1024-dim)
- [x] Claude Sonnet 4.6 LLM working
- [x] OpenSearch collection accessible
- [x] All model IDs updated in notebooks
- [x] Config files created
- [ ] Notebooks 10-23 executed (ready to run)
- [ ] Outputs committed to GitHub (after execution)

## 📝 Next Steps

1. **Execute notebooks**: Use one of the methods above
2. **Review outputs**: Open in Jupyter to verify
3. **Commit to GitHub**: Push notebooks with outputs
4. **Continue to patterns 24-37**: Or improve existing ones

---

**Ready to execute!** Choose your preferred method and start running notebooks.
