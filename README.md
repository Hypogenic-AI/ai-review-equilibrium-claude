# AI Reviewing Equilibrium

This research investigates whether different AI models provide distinct opinions when reviewing academic papers and how multi-model dynamics affect the paper improvement process.

## Key Findings

- **Models differ significantly in ratings**: Gemini 2.0 Flash rates papers ~1 point lower than Claude 3.5 Sonnet (p < 0.0001, Cohen's d = 1.17)
- **High rank correlation despite absolute differences**: Claude-Gemini Spearman r = 0.994 (models agree on which papers are better)
- **All models show positivity bias**: None recommended rejection for any paper, even human-rejected ones
- **Best human calibration**: Gemini is closest to human ratings (mean diff = +0.11)
- **No convergence in dynamics**: Multi-model opinions don't naturally converge across revision iterations

## Quick Start

### Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv add openai httpx numpy pandas matplotlib seaborn scipy scikit-learn tqdm
```

### Run Experiments

```bash
# Run all experiments
python src/run_experiment.py
```

### View Results

Results are saved to:
- `results/experiment_1_results.json` - Multi-model comparison
- `results/experiment_2_dynamics.json` - Revision dynamics
- `figures/` - All visualizations

## Project Structure

```
ai-review-equilibrium-claude/
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Load and prepare PeerRead data
│   ├── review_generator.py   # Multi-model review generation
│   ├── analysis.py           # Statistical analysis functions
│   └── run_experiment.py     # Main experiment runner
├── datasets/
│   ├── PeerRead_raw/         # ICLR 2017 peer review data
│   └── README.md
├── results/                   # Experiment outputs
├── figures/                   # Visualizations
├── papers/                    # Reference papers
├── planning.md               # Research plan
├── REPORT.md                 # Full research report
└── README.md                 # This file
```

## Main Results

### Rating Distributions by Model

| Model | Mean | Std | Accept | Borderline | Reject |
|-------|------|-----|--------|------------|--------|
| GPT-4o-mini | 6.80 | 0.54 | 10 | 5 | 0 |
| Claude 3.5 Sonnet | 7.00 | 0.73 | 11 | 4 | 0 |
| Gemini 2.0 Flash | 6.07 | 0.85 | 2 | 13 | 0 |

### Pairwise Agreement

| Pair | Spearman r | Cohen's d | p-value |
|------|------------|-----------|---------|
| GPT-4 vs Claude | 0.68 | 0.31 | 0.006 |
| GPT-4 vs Gemini | 0.66 | 1.03 | 0.008 |
| Claude vs Gemini | 0.99 | 1.17 | <0.001 |

## Data Sources

- **PeerRead**: 15 papers from ICLR 2017 (10 accepted, 5 rejected)
- **Human reviews**: 2-3 expert reviews per paper with ratings and comments

## API Keys Required

Set these environment variables:
- `OPENAI_API_KEY` - For GPT-4o-mini
- `OPENROUTER_API_KEY` - For Claude and Gemini via OpenRouter

## Full Report

See [REPORT.md](REPORT.md) for the complete research report including:
- Detailed methodology
- Statistical analysis
- Visualizations
- Limitations and future work

## Citation

If you use this research, please cite:
```
AI Reviewing Equilibrium Research, 2025
https://github.com/[repo-path]
```

## License

This research uses the PeerRead dataset under academic use terms.
