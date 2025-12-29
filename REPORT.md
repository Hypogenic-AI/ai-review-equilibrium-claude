# AI Reviewing Equilibrium: Research Report

## 1. Executive Summary

This research investigates whether different AI models provide distinct opinions when reviewing academic papers and how the dynamics of paper improvement change when multiple models are involved in the review process. We conducted experiments using three state-of-the-art LLMs (GPT-4o-mini, Claude 3.5 Sonnet, and Gemini 2.0 Flash) to review 15 papers from the ICLR 2017 PeerRead dataset.

**Key Finding**: Different AI models do provide systematically distinct opinions when reviewing papers. Gemini 2.0 Flash gives significantly lower ratings (mean=6.07) compared to GPT-4o-mini (mean=6.80) and Claude 3.5 Sonnet (mean=7.00), with the difference between Claude and Gemini being statistically significant (p < 0.0001, Cohen's d = 1.17). However, despite different mean ratings, models show moderate-to-high correlation in their relative paper rankings (Spearman r = 0.66-0.99), suggesting they agree on which papers are better or worse, even if they disagree on absolute quality.

**Practical Implication**: The choice of AI model for paper review significantly impacts the outcome. Multi-model reviewing could provide more balanced assessments, though care must be taken to understand each model's calibration bias.

## 2. Goal

### Research Questions
1. **H1 (Distinct Opinions)**: Do different AI models (GPT-4, Claude, Gemini) produce systematically different review feedback on the same papers?
2. **H2 (Multi-Model Dynamics)**: When multiple AI models provide feedback for paper improvement, do the dynamics differ from single-model feedback?

### Why This Matters
- AI-assisted peer review is becoming increasingly common
- Understanding systematic differences between models is crucial for fair and reliable assessments
- Multi-model review systems may offer advantages over single-model approaches

### Gap Addressed
Prior work (AgentReview, MARG) studied multi-agent review using homogeneous model pools. This is the first systematic study of cross-provider multi-model review dynamics.

## 3. Data Construction

### Dataset Description
- **Source**: PeerRead ICLR 2017 (allenai/PeerRead GitHub repository)
- **Total Papers Loaded**: 427 from ICLR 2017
- **Selected for Experiment**: 15 papers (10 accepted, 5 rejected)
- **Selection Criteria**: Papers with 2+ human reviews and abstracts between 100-500 words

### Example Samples

| Paper ID | Title | Accepted | Human Mean Rating |
|----------|-------|----------|-------------------|
| 407 | Attend, Adapt and Transfer: Attentive Deep Architecture... | Yes | 7.0 |
| 383 | Designing Neural Network Architectures using RL | Yes | 6.0 |
| 476 | Do Deep Convolutional Nets Really Need to be Deep... | Yes | 7.33 |
| 352 | Hierarchical compositional feature learning | No | 4.67 |
| 389 | Distributed Transfer Learning for Deep CNNs | No | 3.33 |

### Data Quality
- **Missing Values**: 0% (all papers have complete abstracts and titles)
- **Review Success Rate**: 100% (45/45 reviews successfully generated)
- **Human Review Coverage**: All 15 papers have 2-3 human reviews for comparison

### Preprocessing Steps
1. Loaded papers from PeerRead JSON format
2. Filtered to papers with 2+ non-meta reviews with RECOMMENDATION scores
3. Selected balanced sample of accepted/rejected papers
4. Extracted human review statistics for comparison baseline

## 4. Experiment Description

### Methodology

#### High-Level Approach
We conducted two experiments:
1. **Experiment 1**: Multi-model review comparison - each of 3 models reviews 15 papers
2. **Experiment 2**: Paper improvement dynamics - simulate revisions and track rating changes

#### Why This Method?
- Direct comparison allows measuring absolute and relative differences
- Using real papers from PeerRead enables comparison with human reviewers
- Dynamics experiment tests how multi-model feedback affects paper improvement

### Implementation Details

#### Tools and Libraries
- Python 3.12.2
- openai 2.14.0 (for GPT-4o-mini)
- httpx 0.28.1 (for OpenRouter API)
- scipy 1.16.3 (statistical tests)
- matplotlib 3.10.8, seaborn 0.13.2 (visualization)

#### Models Used
| Model ID | Provider | Display Name | Purpose |
|----------|----------|--------------|---------|
| gpt4 | OpenAI | GPT-4o-mini | Primary review model |
| claude | OpenRouter | Claude 3.5 Sonnet | Comparison model |
| gemini | OpenRouter | Gemini 2.0 Flash | Comparison model |

#### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| temperature | 0 | Reproducibility |
| max_tokens | 2000 | Sufficient for reviews |
| random_seed | 42 | Reproducibility |

#### Review Prompt
Standardized ICLR-style review prompt requesting:
- Summary (2-3 sentences)
- Strengths (3-5 items)
- Weaknesses (3-5 items)
- Questions (2-3 items)
- Rating (1-10 scale)
- Confidence (1-5 scale)
- Recommendation (accept/reject/borderline)

### Experimental Protocol

#### Reproducibility Information
- **Number of API Calls**: 45 (Experiment 1) + 30 (Experiment 2) = 75
- **Random Seeds**: 42 (numpy, random)
- **Hardware**: CPU-based (API calls only)
- **Execution Time**: ~10 minutes total

#### Evaluation Metrics

| Metric | Purpose | Interpretation |
|--------|---------|----------------|
| Mean Rating | Central tendency | Higher = more positive |
| Spearman Correlation | Rank agreement | >0.7 = strong agreement |
| Weighted Kappa | Category agreement | >0.6 = substantial |
| Cohen's d | Effect size | >0.8 = large |
| Paired t-test p | Statistical significance | <0.05 = significant |

### Raw Results

#### Per-Model Rating Statistics

| Model | Mean | Std | Min | Max | Median | Accept | Borderline | Reject |
|-------|------|-----|-----|-----|--------|--------|------------|--------|
| GPT-4o-mini | 6.80 | 0.54 | 6 | 8 | 7.0 | 10 | 5 | 0 |
| Claude 3.5 Sonnet | 7.00 | 0.73 | 6 | 8 | 7.0 | 11 | 4 | 0 |
| Gemini 2.0 Flash | 6.07 | 0.85 | 5 | 8 | 6.0 | 2 | 13 | 0 |

#### Pairwise Comparison Results

| Comparison | Spearman r | p-value | Kappa | Mean Diff | Cohen's d |
|------------|------------|---------|-------|-----------|-----------|
| GPT-4 vs Claude | 0.675 | 0.006 | 0.659 | -0.20 | -0.31 |
| GPT-4 vs Gemini | 0.658 | 0.008 | 0.234 | +0.73 | 1.03 |
| Claude vs Gemini | 0.994 | <0.001 | 0.234 | +0.93 | 1.17 |

#### AI vs Human Review Comparison

| Model | Mean Diff from Human | Correlation with Human | p-value |
|-------|----------------------|------------------------|---------|
| GPT-4o-mini | +0.84 | 0.458 | 0.086 |
| Claude 3.5 Sonnet | +1.04 | 0.815 | <0.001 |
| Gemini 2.0 Flash | +0.11 | 0.823 | <0.001 |

#### Content Analysis

| Model | Avg Strengths | Avg Weaknesses | Avg Questions | S/W Ratio |
|-------|---------------|----------------|---------------|-----------|
| GPT-4o-mini | 3.67 | 3.67 | 3.0 | 1.00 |
| Claude 3.5 Sonnet | 4.27 | 4.00 | 3.0 | 1.07 |
| Gemini 2.0 Flash | 4.33 | 4.53 | 3.0 | 0.98 |

### Visualizations

![Rating Distributions](figures/rating_distributions.png)
*Figure 1: Rating distributions by AI model showing Gemini's lower mean ratings*

![Pairwise Correlation](figures/pairwise_correlation.png)
*Figure 2: Spearman correlation heatmap between models*

![Recommendation Distribution](figures/recommendation_distribution.png)
*Figure 3: Recommendation distribution showing Claude's tendency toward acceptance*

![Human Comparison](figures/human_comparison.png)
*Figure 4: AI model ratings vs human reviewer ratings*

![Content Analysis](figures/content_analysis.png)
*Figure 5: Average number of strengths, weaknesses, and questions per review*

![Dynamics Evolution](figures/dynamics_evolution.png)
*Figure 6: Rating changes across revision iterations*

## 5. Result Analysis

### Key Findings

#### Finding 1: Models Show Significant Rating Differences
- **Gemini 2.0 Flash rates papers significantly lower** than both GPT-4o-mini and Claude 3.5 Sonnet
- The difference between Claude (mean=7.0) and Gemini (mean=6.07) is nearly a full point
- Statistical significance: p < 0.0001 (Claude vs Gemini), p = 0.001 (GPT-4 vs Gemini)
- Effect sizes are large: Cohen's d = 1.17 (Claude vs Gemini), 1.03 (GPT-4 vs Gemini)

#### Finding 2: High Rank Correlation Despite Absolute Differences
- Despite different mean ratings, models agree substantially on relative quality
- Claude-Gemini Spearman correlation: r = 0.994 (near-perfect rank agreement)
- GPT-4-Claude and GPT-4-Gemini: r ≈ 0.67 (moderate-strong agreement)
- This suggests models agree on *which* papers are better, but differ on *how good* they are

#### Finding 3: Different Calibration vs Human Reviewers
- **Gemini is best calibrated** to human reviewers (mean diff = +0.11)
- **Claude and GPT-4 are overly positive** (mean diff = +1.04 and +0.84 respectively)
- However, Claude and Gemini have higher **correlation** with humans (r ≈ 0.82)
- GPT-4 shows weaker correlation with humans (r = 0.46, p = 0.086)

#### Finding 4: Content Style Differences
- **Gemini provides more weaknesses** (4.53 avg) than strengths (4.33 avg)
- **Claude is slightly strength-biased** (4.27 strengths vs 4.00 weaknesses)
- **GPT-4 is balanced** (3.67 each)
- All models provide similar number of questions (3.0)

#### Finding 5: Limited Convergence in Dynamics Experiment
- Over 2 revision iterations, model opinions **did not converge**
- Rating standard deviation remained constant (0.47) across iterations
- GPT-4 showed slight positive trend (mean change = +0.89)
- Claude and Gemini remained stable (mean change ≈ -0.1)

### Hypothesis Testing Results

#### H1: Different models provide distinct opinions
- **SUPPORTED**: Statistically significant differences in mean ratings (p < 0.05)
- Largest effect: Claude vs Gemini (Cohen's d = 1.17, large effect)
- Kappa agreement on rating categories: 0.23-0.66 (fair to substantial)
- Models differ in calibration but agree on relative rankings

#### H2: Multi-model dynamics differ from single-model
- **PARTIALLY SUPPORTED**: Models maintain different opinions across revisions
- No convergence observed after 2 iterations
- Different models respond differently to revisions (GPT-4 more positive, others stable)
- Limited iterations prevent stronger conclusions

### Comparison to Literature

Our findings align with prior work:
- **OpenReviewer paper**: Found general LLMs are "overly positive" - we confirm this for GPT-4 and Claude
- **LLM-as-Judge survey**: Identified model-specific biases - we quantify these differences
- **AgentReview**: Showed reviewer biases affect outcomes - we extend this to cross-provider differences

### Surprises and Insights

1. **Near-perfect Claude-Gemini rank correlation** (r=0.994) despite 1-point mean difference - suggests these models may share underlying evaluation criteria

2. **GPT-4's weaker human correlation** is unexpected given its general reputation - possibly due to its tendency toward positive reviews

3. **No rejection recommendations from any model** - all 15 papers received borderline or accept from all models, even for papers humans rejected

### Error Analysis

- **1 parsing failure** in Experiment 2 (GPT-4 failed to produce valid JSON once)
- All models show **positivity bias** - no reject recommendations despite 5 rejected papers in dataset
- Models may be influenced by **abstract-only information** (no access to full paper methodology/results)

### Limitations

1. **Sample Size**: Only 15 papers - larger sample needed for robust conclusions
2. **Abstract-Only**: Reviews based only on abstracts, not full papers
3. **2017 Papers**: Papers may be in training data of newer models
4. **Limited Iterations**: Only 2 revision iterations in dynamics experiment
5. **No Specialized Models**: Did not include fine-tuned review models like OpenReviewer
6. **Cost-Efficient Models**: Used GPT-4o-mini instead of full GPT-4/GPT-5 for cost reasons

## 6. Conclusions

### Summary

This research demonstrates that **different AI models provide systematically distinct opinions when reviewing academic papers**. Gemini 2.0 Flash is significantly more critical (lower ratings) than GPT-4o-mini and Claude 3.5 Sonnet, though all models show strong rank correlation. All tested models exhibit positivity bias compared to human reviewers, with Gemini being best calibrated.

The dynamics experiment suggests that multi-model disagreement persists across revision iterations, with no evidence of natural convergence. This has implications for multi-model review systems: simply averaging opinions may not produce balanced outcomes.

### Implications

#### Practical Implications
- **Model selection matters**: Different models will produce different review outcomes
- **Multi-model averaging requires calibration**: Raw score averaging would be dominated by more positive models
- **Gemini may be preferred for critical reviews**: Lower positivity bias, closer to human calibration

#### Theoretical Implications
- AI models have learned different implicit standards for paper quality
- High rank correlation suggests shared underlying quality assessment criteria
- Absolute calibration differs significantly across model providers

### Confidence in Findings
- **High confidence**: Models differ in mean ratings (p < 0.001, large effect sizes)
- **High confidence**: Models agree on relative paper rankings (r > 0.65)
- **Moderate confidence**: Positivity bias exists (limited to 15 papers)
- **Low confidence**: Dynamics conclusions (only 2 iterations, 5 papers)

## 7. Next Steps

### Immediate Follow-ups
1. **Increase sample size** to 100+ papers for robust statistical power
2. **Test full GPT-4/5 and Claude Opus** instead of mini versions
3. **Add specialized review models** (OpenReviewer) for comparison
4. **Extend dynamics to 5+ iterations** to observe convergence patterns

### Alternative Approaches
- Use full paper PDFs instead of abstracts only
- Test prompt variations to reduce positivity bias
- Implement calibration normalization for multi-model systems

### Broader Extensions
- Study bias patterns across different paper domains (ML, NLP, vision)
- Investigate whether models can detect AI-generated papers
- Design optimal multi-model aggregation strategies

### Open Questions
1. Why do Claude and Gemini have near-perfect rank correlation despite different calibrations?
2. Can prompting strategies eliminate positivity bias?
3. What drives the lack of convergence in multi-model revision cycles?

---

## References

### Papers
1. Jin et al. (2024). AgentReview: Exploring Peer Review Dynamics with LLM Agents. EMNLP.
2. D'Arcy et al. (2024). MARG: Multi-Agent Review Generation for Scientific Papers. arXiv.
3. Idahl & Ahmadi (2024). OpenReviewer: A Specialized LLM for Critical Paper Reviews. arXiv.
4. Kang et al. (2018). PeerRead: A Dataset of Peer Reviews. NAACL.
5. Gu et al. (2024). A Survey on LLM-as-a-Judge. arXiv.

### Datasets
- PeerRead ICLR 2017: https://github.com/allenai/PeerRead

### Tools
- OpenAI API (GPT-4o-mini)
- OpenRouter API (Claude 3.5 Sonnet, Gemini 2.0 Flash)
- scipy, matplotlib, seaborn for analysis and visualization

---

*Report generated: 2025-12-29*
*Research conducted using automated LLM-based experiments*
