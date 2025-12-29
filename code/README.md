# Cloned Repositories

This directory contains code repositories relevant to the AI Reviewing Equilibrium research project.

## Repository 1: AgentReview

- **URL**: https://github.com/Ahren09/AgentReview
- **Purpose**: LLM-based peer review simulation framework (EMNLP 2024)
- **Location**: `code/AgentReview/`
- **Key Features**:
  - Simulates complete peer review process with LLM agents
  - Multiple reviewer characteristics (commitment, intention, knowledgeability)
  - Generates reviews, rebuttals, meta-reviews, and final decisions
  - 53,800+ generated review documents

### Key Files
- `run_paper_review_cli.py` - Main script to run paper reviews
- `run_paper_decision_cli.py` - Script to make paper decisions
- `agentreview/` - Core framework code
- `notebooks/` - Analysis notebooks
- `requirements.txt` - Python dependencies

### Usage
```bash
cd code/AgentReview
pip install -r requirements.txt
python run_paper_review_cli.py --help
```

### Notes
- Requires OpenAI API key for LLM calls
- Download data from Dropbox links in repository README
- Supports various reviewer bias configurations

---

## Repository 2: MARG Reviewer

- **URL**: https://github.com/allenai/marg-reviewer
- **Purpose**: Multi-Agent Review Generation for scientific papers
- **Location**: `code/marg-reviewer/`
- **Key Features**:
  - Multi-agent architecture with leader, worker, and expert agents
  - Handles papers longer than LLM context window
  - Specialized feedback generation (experiments, clarity, impact)
  - Web interface for user study

### Key Files
- `api/` - API endpoints for review generation
- `review_worker/` - Core review worker code
- `app_data/` - Application data
- `docker-compose.yaml` - Docker setup

### Usage
```bash
cd code/marg-reviewer
# See README for Docker setup
docker-compose up
```

### Notes
- Uses GPT-4 for review generation
- Requires API keys for OpenAI
- Includes user study interface

---

## Repository 3: OpenReview Data Crawler

- **URL**: https://github.com/hughplay/ICLR2024-OpenReviewData
- **Purpose**: Crawl and visualize ICLR data from OpenReview
- **Location**: `code/OpenReviewData/`
- **Key Features**:
  - Fast parallel crawling via OpenReview API
  - Data visualization tools
  - Pre-crawled data for ICLR 2024

### Key Files
- Check repository README for main scripts
- Includes visualization notebooks

### Usage
```bash
cd code/OpenReviewData
pip install -r requirements.txt
# See README for crawling instructions
```

### Notes
- Useful for collecting fresh OpenReview data
- Rate limiting applies to API calls

---

## Recommendations for Experiments

### For Multi-Model Review Dynamics (Primary Research Focus)

**Use AgentReview framework** to:
1. Configure multiple LLM agents with different "personalities" (e.g., GPT-4 vs Claude vs Gemini)
2. Have each model review the same paper independently
3. Simulate the reviewer-author discussion phase with different models
4. Analyze how different model combinations affect final decisions

### For Review Quality Comparison

**Use MARG framework** to:
1. Generate reviews with different model configurations
2. Compare multi-agent vs single-agent review quality
3. Evaluate specialized expert agents for different aspects

### For Data Collection

**Use OpenReview Crawler** to:
1. Collect recent ICLR/NeurIPS review data
2. Build evaluation datasets with ground truth human reviews
3. Compare LLM reviews against human baselines

---

## Dependencies Summary

All repositories require:
- Python 3.8+
- OpenAI API access (GPT-4)
- Additional dependencies in respective requirements.txt files

Install common dependencies:
```bash
pip install openai anthropic openreview-py
```
