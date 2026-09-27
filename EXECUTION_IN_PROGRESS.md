# ⏳ Notebook Execution In Progress

## Status: RUNNING

Batch execution of notebooks 10-23 is currently running in the background.

**Started**: Just now
**Process ID**: Check with `ps aux | grep jupyter`
**Expected Duration**: 1-2 hours total
**Estimated Cost**: ~$3-7

## Execution Order

Notebooks are executing in this order (easiest to hardest):

1. ✓ 20_Iterative_RAG_AWS.ipynb (Running now)
2. ⏳ 21_Few_Shot_RAG_AWS.ipynb
3. ⏳ 22_Hierarchical_RAG_AWS.ipynb
4. ⏳ 23_Parent_Child_RAG_AWS.ipynb
5. ⏳ 16_Chain_of_Thought_RAG_AWS.ipynb
6. ⏳ 12_Agentic_RAG_AWS.ipynb
7. ⏳ 17_ReAct_RAG_AWS.ipynb
8. ⏳ 13_Corrective_RAG_AWS.ipynb
9. ⏳ 14_Self_RAG_AWS.ipynb
10. ⏳ 10_Recursive_RAG_AWS.ipynb
11. ⏳ 15_Tree_of_Thoughts_RAG_AWS.ipynb
12. ⏳ 19_Ensemble_RAG_AWS.ipynb
13. ⏳ 18_Memory_Augmented_RAG_AWS.ipynb
14. ⏳ 11_Multimodal_RAG_AWS.ipynb

## How to Monitor Progress

### Check Progress
```bash
cd /workshop/rag_pattern_notebooks
./check_progress.sh
```

### View Live Output
```bash
tail -f /tmp/claude-1001/-workshop/*/tasks/b3t73278a.output
```

### Check Individual Notebook Logs
```bash
cd /workshop/rag_pattern_notebooks
ls execution_*.log
tail -50 execution_20_Iterative_RAG_AWS.log
```

## What's Happening

Each notebook goes through these steps:

1. **Import** aws_utils modules
2. **Initialize** AWS services (Bedrock, OpenSearch)
3. **Create** sample documents
4. **Embed** documents using Titan
5. **Index** in OpenSearch
6. **Execute** pattern-specific logic
7. **Generate** answers using Claude Sonnet 4.6
8. **Save** outputs in notebook cells

## After Completion

When all notebooks finish, you'll need to:

### 1. Verify Execution
```bash
cd /workshop/rag_pattern_notebooks
./check_progress.sh
# Should show 14/14 completed
```

### 2. Review Outputs
```bash
# Open in Jupyter to see results
source venv/bin/activate
jupyter notebook
```

### 3. Commit to GitHub
```bash
git add aws_notebooks/*.ipynb
git commit -m "Add execution outputs for notebooks 10-23

All notebooks executed successfully with AWS:
- Model: us.anthropic.claude-sonnet-4-6
- Embeddings: amazon.titan-embed-text-v2:0
- Region: us-west-2
- OpenSearch: qrm9kbjh7wmnpa99ee2b.us-west-2.aoss.amazonaws.com

Outputs saved in all cells


git push origin main
```

## If Something Goes Wrong

### Execution Stops
Check logs:
```bash
cat execution_*.log | grep -i "error\|failed"
```

### Notebook Fails
Re-run individual notebook:
```bash
source venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  aws_notebooks/XX_Pattern_Name_AWS.ipynb
```

### Out of Memory
Reduce concurrent processes or restart with more memory.

### API Rate Limits
Bedrock has rate limits. If hit, wait a few minutes and retry.

## Estimated Timeline

| Time | Status |
|------|--------|
| 0-10 min | First 3-4 notebooks complete |
| 10-30 min | Next 4-5 notebooks complete |
| 30-60 min | Complex notebooks (Tree, Ensemble) |
| 60-90 min | Memory-heavy notebooks |
| 90-120 min | All complete |

## Cost Tracking

Watch costs in AWS Cost Explorer:
- Bedrock: ~$2.50-6.50
- OpenSearch: ~$0.24/hour
- Total: ~$3-7

## Current Configuration

All notebooks are using:
```python
AWS_REGION = 'us-west-2'
EMBEDDING_MODEL = 'amazon.titan-embed-text-v2:0'
LLM_MODEL = 'us.anthropic.claude-sonnet-4-6'
OPENSEARCH_ENDPOINT = 'https://qrm9kbjh7wmnpa99ee2b.us-west-2.aoss.amazonaws.com'
```

## Files Being Created

- `execution_*.log` - Execution logs for each notebook
- Updated `.ipynb` files with outputs in cells
- Temporary OpenSearch indexes (auto-cleaned)

---

**Status**: ⏳ IN PROGRESS
**Last Updated**: Just now
**Check Again In**: 10-15 minutes
