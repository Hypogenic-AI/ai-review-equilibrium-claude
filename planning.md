# Research Plan: AI Reviewing Equilibrium

## Research Question

**Do different AI models provide distinct opinions when reviewing academic papers, and how do the dynamics of paper improvement change when multiple models are involved in the review process?**

This research tests two hypotheses:
1. **H1 (Distinct Opinions)**: Different AI models (GPT-4, Claude, Gemini) produce systematically different review feedback on the same papers
2. **H2 (Multi-Model Dynamics)**: When multiple AI models provide feedback for paper improvement, the dynamics differ from single-model feedback (convergence, divergence, or equilibrium effects)

## Background and Motivation

AI-assisted peer review has emerged as a potential solution to the growing challenges in academic publishing. However, critical questions remain unanswered:

1. **Model-Specific Biases**: Literature (OpenReviewer, LLM-as-Judge survey) suggests different models have distinct biases - general-purpose LLMs are overly positive while specialized models are more critical
2. **Multi-Agent Dynamics**: AgentReview and MARG show multi-agent systems improve review quality, but these use homogeneous model pools
3. **Equilibrium Effects**: No systematic study exists on what happens when heterogeneous AI models (different providers) participate in review processes

**Gap Addressed**: This research provides the first systematic study of cross-provider multi-model review dynamics.

## Hypothesis Decomposition

### H1: Distinct Opinions
- **H1a**: Models differ in numerical ratings (recommendation scores)
- **H1b**: Models differ in review sentiment (positive/negative tone)
- **H1c**: Models focus on different aspects (methodology, clarity, novelty, etc.)
- **H1d**: Inter-model agreement (Cohen's Kappa) is significantly lower than intra-model consistency

### H2: Multi-Model Dynamics
- **H2a**: Multi-model feedback leads to different revision priorities than single-model feedback
- **H2b**: Papers revised based on multi-model feedback show convergence/divergence patterns
- **H2c**: Certain models "dominate" in consensus formation

## Proposed Methodology

### Approach

We conduct a three-part study:

**Part 1: Multi-Model Review Comparison**
- Have GPT-4, Claude, and Gemini review the same set of papers
- Measure agreement/disagreement across multiple dimensions
- Compare against human baseline (from PeerRead)

**Part 2: Review Content Analysis**
- Analyze qualitative differences in review content
- Identify aspect coverage patterns
- Measure sentiment and critique specificity

**Part 3: Paper Improvement Dynamics**
- Simulate paper improvement based on multi-model feedback
- Track convergence/divergence of model opinions over iterations
- Study equilibrium formation

### Experimental Steps

1. **Data Selection** (15 papers from PeerRead ICLR 2017)
   - Mix of accepted (10) and rejected (5) papers
   - Papers with 3+ human reviews for comparison
   - Papers with abstracts 150-500 words for consistency

2. **Review Generation**
   - Generate reviews from GPT-4.1, Claude Sonnet 4.5, Gemini 2.5 Pro
   - Use standardized review prompt (conference format)
   - Extract: overall rating (1-10), strengths, weaknesses, questions, recommendation

3. **Quantitative Analysis**
   - Inter-model agreement (Cohen's Kappa for ratings)
   - Rating distribution comparison
   - Correlation with human reviews

4. **Qualitative Analysis**
   - Topic modeling on review content
   - Aspect coverage analysis (methodology, clarity, novelty, impact)
   - Sentiment analysis

5. **Improvement Dynamics**
   - Select subset of papers
   - Generate revision suggestions from each model
   - Create synthetic "revised" abstracts incorporating feedback
   - Re-review with all models
   - Measure opinion convergence/divergence

### Baselines

| Baseline | Description |
|----------|-------------|
| Human Reviews | From PeerRead dataset |
| Single-Model GPT-4 | Standard single-model review |
| Single-Model Claude | Standard single-model review |
| Single-Model Gemini | Standard single-model review |
| Random | Random rating baseline |

### Evaluation Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| Cohen's Kappa | Pairwise inter-rater agreement | Measure model agreement |
| Fleiss' Kappa | Multi-rater agreement | Multi-model consensus |
| Pearson/Spearman Correlation | Rating correlation with humans | Accuracy vs ground truth |
| Rating Distribution (KL Divergence) | Compare rating distributions | Calibration |
| Aspect Coverage | % of aspects addressed | Comprehensiveness |
| Sentiment Score | Positive/negative tone | Review tone analysis |
| Jaccard Similarity | Overlap in raised issues | Content similarity |

### Statistical Analysis Plan

- **Significance Level**: α = 0.05
- **Tests**:
  - Paired t-tests for rating comparisons (if normally distributed)
  - Wilcoxon signed-rank test for non-normal distributions
  - Chi-square test for categorical comparisons (accept/reject)
  - ANOVA for comparing >2 groups
- **Effect Sizes**: Cohen's d for pairwise comparisons
- **Multiple Comparisons**: Bonferroni correction when applicable

## Expected Outcomes

### If H1 is Supported (Distinct Opinions):
- Inter-model Kappa < 0.6 (substantial disagreement)
- Different models focus on different review aspects
- Rating distributions differ significantly (p < 0.05)

### If H1 is Not Supported:
- High inter-model agreement (Kappa > 0.8)
- Similar content and ratings across models
- Suggests AI reviews converge regardless of provider

### If H2 is Supported (Dynamic Effects):
- Multi-model revision suggestions differ from single-model
- Convergence or divergence patterns emerge over iterations
- Evidence of "dominant" models in consensus formation

## Timeline and Milestones

| Phase | Tasks | Estimated Duration |
|-------|-------|-------------------|
| 1 | Planning & Setup | 20 min |
| 2 | Data Preparation | 20 min |
| 3 | Review Generation | 30 min |
| 4 | Quantitative Analysis | 30 min |
| 5 | Qualitative Analysis | 20 min |
| 6 | Dynamics Experiment | 30 min |
| 7 | Documentation | 20 min |

## Potential Challenges

| Challenge | Mitigation |
|-----------|------------|
| API rate limits | Batch requests, add delays |
| Model availability | Use OpenRouter as fallback |
| Cost constraints | Limit to 15 papers |
| Review variability | Multiple samples, temperature=0 |
| Ground truth ambiguity | Use multiple metrics |

## Success Criteria

1. **Minimum**: Complete experiments for 15 papers across 3 models
2. **Adequate**: Statistical analysis with significance testing
3. **Complete**: Full analysis including dynamics experiment with visualizations
4. **Excellent**: Novel insights about AI review equilibrium with actionable implications

## Resource Utilization

### From Pre-Gathered Resources

| Resource | Usage |
|----------|-------|
| PeerRead Dataset | Source of papers and human reviews |
| Literature Review | Methodology guidance, baseline comparison |
| AgentReview Code | Potential adaptation for multi-agent simulation |

### APIs Required

| API | Model | Purpose |
|-----|-------|---------|
| OpenAI | GPT-4.1 | Review generation |
| OpenRouter | Claude, Gemini | Review generation |

## Ethical Considerations

- Using publicly available PeerRead data (academic use)
- No manipulation of real peer review processes
- Results are exploratory research, not recommendations for policy
