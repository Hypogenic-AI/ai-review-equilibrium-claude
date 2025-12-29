# Resources Catalog

## Summary

This document catalogs all resources gathered for the AI Reviewing Equilibrium research project. The project investigates how different AI models provide distinct opinions when reviewing papers and how multi-model dynamics affect paper improvement processes.

**Resources Collected:**
- Papers: 9 PDFs downloaded
- Datasets: 2 datasets downloaded locally
- Repositories: 3 code repositories cloned

---

## Papers

**Total papers downloaded: 9**

| # | Title | Authors | Year | File | Key Contribution |
|---|-------|---------|------|------|------------------|
| 1 | A Dataset of Peer Reviews (PeerRead) | Kang et al. | 2018 | papers/1804.09635_PeerRead.pdf | First public peer review dataset |
| 2 | MARG: Multi-Agent Review Generation | D'Arcy et al. | 2024 | papers/2401.04259_MARG.pdf | Multi-agent review framework |
| 3 | AgentReview: Exploring Peer Review Dynamics | Jin et al. | 2024 | papers/2406.12708_AgentReview.pdf | LLM peer review simulation |
| 4 | AI-Driven Review Systems | Various | 2024 | papers/2408.10365_AI_Driven_Review.pdf | Scalable AI reviews |
| 5 | Survey on LLM-as-a-Judge | Gu et al. | 2024 | papers/2411.15594_LLM_as_Judge_Survey.pdf | LLM evaluation methods |
| 6 | LLMs-as-Judges Survey | Li et al. | 2024 | papers/2412.05579_LLM_as_Judges_Survey.pdf | Comprehensive LLM evaluation survey |
| 7 | OpenReviewer | Idahl & Ahmadi | 2024 | papers/2412.11948_OpenReviewer.pdf | Fine-tuned review model |
| 8 | LLMs for ASPR: A Survey | Zhuang et al. | 2025 | papers/2501.10326_LLM_ASPR_Survey.pdf | Automated paper review survey |
| 9 | ReviewAgents | Gao et al. | 2025 | papers/2503.08506_ReviewAgents.pdf | Multi-role review framework |

See `papers/README.md` for detailed descriptions.

---

## Datasets

**Total datasets downloaded: 2** (locally available)

| # | Name | Source | Size | Task | Location | Notes |
|---|------|--------|------|------|----------|-------|
| 1 | PeerRead | GitHub | 14.7K papers, 10.7K reviews | Review analysis | datasets/PeerRead_raw/ | Complete with train/dev/test splits |
| 2 | SEA OpenReview | HuggingFace | 8K+ samples | Paper analysis | datasets/sea_openreview/ | Paper IDs format |

### Additional Datasets (Not Downloaded, Instructions Provided)

| Name | Source | Access Method |
|------|--------|---------------|
| AgentReview Simulation | Dropbox | Clone repo, download from links |
| OpenReview ICLR/NeurIPS | API | Use openreview-py package |
| Intel AI-Peer-Review | HuggingFace | Schema issues, use API directly |

See `datasets/README.md` for download instructions and usage details.

---

## Code Repositories

**Total repositories cloned: 3**

| # | Name | URL | Purpose | Location | Notes |
|---|------|-----|---------|----------|-------|
| 1 | AgentReview | github.com/Ahren09/AgentReview | LLM peer review simulation | code/AgentReview/ | EMNLP 2024, requires GPT-4 API |
| 2 | MARG Reviewer | github.com/allenai/marg-reviewer | Multi-agent review generation | code/marg-reviewer/ | Docker-based, includes user study interface |
| 3 | OpenReview Data | github.com/hughplay/ICLR2024-OpenReviewData | Crawl OpenReview data | code/OpenReviewData/ | Fast parallel API crawling |

See `code/README.md` for usage instructions.

---

## Resource Gathering Notes

### Search Strategy

1. **Initial Search**: Web search for recent arXiv papers on LLM peer review, multi-agent review systems
2. **Citation Following**: Identified key papers from survey references
3. **Code Discovery**: Found GitHub repos from paper links and Papers with Code
4. **Dataset Identification**: Extracted dataset information from paper methods sections

### Search Keywords Used
- "LLM peer review"
- "AI paper review"
- "automated scientific review"
- "LLM as judge"
- "multi-agent review"
- "AgentReview"
- "OpenReviewer"
- "MARG multi-agent"
- "PeerRead dataset"

### Selection Criteria

Papers selected based on:
1. **Relevance**: Directly addresses AI/LLM peer review
2. **Recency**: Preference for 2024-2025 publications
3. **Multi-agent focus**: Papers studying multiple reviewers/models
4. **Code availability**: Preference for papers with available implementations
5. **Dataset contribution**: Papers introducing useful datasets

### Challenges Encountered

1. **HuggingFace Dataset Issues**: Some datasets (Intel Labs, allenai/peer_read) have deprecated loading scripts or schema issues
2. **Large Dataset Sizes**: AgentReview full data requires Dropbox downloads (~GB scale)
3. **API Rate Limits**: OpenReview API has rate limiting for fresh data collection

### Gaps and Workarounds

| Gap | Workaround |
|-----|------------|
| No cross-model review dataset | Use PeerRead as base + generate reviews with multiple models |
| Limited recent human reviews | Use OpenReview API to collect fresh data |
| No equilibrium dynamics study | Design experiments using AgentReview framework |

---

## Recommendations for Experiment Design

Based on gathered resources, I recommend:

### 1. Primary Dataset(s)

**PeerRead (ICLR 2017)** - Best choice because:
- Multiple reviewers per paper (averaging 3+)
- Rich review data with aspect scores (originality, clarity)
- Accept/reject ground truth
- Enables comparison of LLM reviews vs human reviews

**OpenReview API (ICLR 2024)** - For fresh data:
- Most recent review data available
- Can collect papers and reviews programmatically
- Useful for evaluation on contemporary papers

### 2. Baseline Methods

| Baseline | Description | Implementation |
|----------|-------------|----------------|
| GPT-4 Review | Single model review | OpenAI API |
| Claude Review | Single model review | Anthropic API |
| Gemini Review | Single model review | Google API |
| Human Review | Expert baseline | From PeerRead |
| AgentReview (same model) | Multi-agent same provider | AgentReview repo |
| MARG | Multi-agent specialized | MARG repo |

### 3. Evaluation Metrics

| Metric | Purpose | Computation |
|--------|---------|-------------|
| Inter-model Agreement (Kappa) | Measure model consensus | Cohen's/Fleiss' Kappa |
| Correlation with Human | Accuracy validation | Spearman correlation |
| Comment Specificity | Review quality | Manual/LLM scoring |
| Rating Distribution Match | Calibration | KL divergence |
| Decision Agreement | Reliability | % agreement |

### 4. Code to Adapt/Reuse

| Repository | What to Use | Adaptation Needed |
|------------|-------------|-------------------|
| AgentReview | Full simulation pipeline | Add Claude/Gemini agents |
| MARG | Multi-agent communication | Extend to cross-model discussion |
| OpenReviewData | Data collection scripts | Use for fresh evaluation data |

### 5. Experimental Design

**Experiment 1: Single-Model Comparison**
```
For each paper in PeerRead test set:
    1. Get GPT-4 review
    2. Get Claude review
    3. Get Gemini review
    4. Compare: ratings, comments, recommendations
    5. Measure: inter-model agreement, deviation from human
```

**Experiment 2: Multi-Model Discussion Simulation**
```
Extend AgentReview:
    1. Assign papers to 3 reviewers from different models
    2. Run simulation with cross-model discussion
    3. Measure: consensus formation, final decision quality
    4. Compare: same-model vs cross-model outcomes
```

**Experiment 3: Paper Improvement Dynamics**
```
1. Take paper with known reviews
2. Generate revision suggestions from each model
3. Apply revisions iteratively
4. Re-review with all models
5. Track: convergence vs divergence of opinions
```

---

## File Manifest

```
ai-review-equilibrium-claude/
├── papers/
│   ├── README.md
│   ├── 1804.09635_PeerRead.pdf
│   ├── 2401.04259_MARG.pdf
│   ├── 2406.12708_AgentReview.pdf
│   ├── 2408.10365_AI_Driven_Review.pdf
│   ├── 2411.15594_LLM_as_Judge_Survey.pdf
│   ├── 2412.05579_LLM_as_Judges_Survey.pdf
│   ├── 2412.11948_OpenReviewer.pdf
│   ├── 2501.10326_LLM_ASPR_Survey.pdf
│   └── 2503.08506_ReviewAgents.pdf
├── datasets/
│   ├── .gitignore
│   ├── README.md
│   ├── PeerRead_raw/          # ~1.6GB
│   └── sea_openreview/        # ~145KB
├── code/
│   ├── README.md
│   ├── AgentReview/           # ~6MB
│   ├── marg-reviewer/         # ~19MB
│   └── OpenReviewData/        # ~75MB
├── literature_review.md
└── resources.md (this file)
```

---

## Next Steps for Experiment Runner

1. **Environment Setup**
   - Install dependencies from each repo's requirements.txt
   - Set up API keys for OpenAI, Anthropic, Google
   - Test API access with simple prompts

2. **Data Preparation**
   - Load PeerRead ICLR 2017 test set
   - Extract papers and human reviews
   - Create evaluation splits

3. **Baseline Implementation**
   - Implement single-model review generation
   - Add multi-model AgentReview configuration
   - Set up MARG with different models

4. **Evaluation Pipeline**
   - Implement agreement metrics
   - Set up human review comparison
   - Create visualization for review dynamics

5. **Experiment Execution**
   - Run single-model comparisons
   - Run multi-model simulations
   - Analyze equilibrium dynamics
