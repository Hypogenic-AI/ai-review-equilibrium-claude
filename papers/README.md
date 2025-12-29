# Downloaded Papers

This directory contains academic papers relevant to the AI Reviewing Equilibrium research project.

## Paper List

| # | File | Title | Authors | Year | Key Topic |
|---|------|-------|---------|------|-----------|
| 1 | [1804.09635_PeerRead.pdf](1804.09635_PeerRead.pdf) | A Dataset of Peer Reviews (PeerRead) | Kang et al. | 2018 | Foundational peer review dataset |
| 2 | [2401.04259_MARG.pdf](2401.04259_MARG.pdf) | MARG: Multi-Agent Review Generation for Scientific Papers | D'Arcy et al. | 2024 | Multi-agent review framework |
| 3 | [2406.12708_AgentReview.pdf](2406.12708_AgentReview.pdf) | AgentReview: Exploring Peer Review Dynamics with LLM Agents | Jin et al. | 2024 | LLM-based peer review simulation |
| 4 | [2408.10365_AI_Driven_Review.pdf](2408.10365_AI_Driven_Review.pdf) | AI-Driven Review Systems | Various | 2024 | Scalable AI-based reviews |
| 5 | [2411.15594_LLM_as_Judge_Survey.pdf](2411.15594_LLM_as_Judge_Survey.pdf) | A Survey on LLM-as-a-Judge | Gu et al. | 2024 | LLM evaluation methods survey |
| 6 | [2412.05579_LLM_as_Judges_Survey.pdf](2412.05579_LLM_as_Judges_Survey.pdf) | LLMs-as-Judges: A Comprehensive Survey | Li et al. | 2024 | LLM-based evaluation comprehensive survey |
| 7 | [2412.11948_OpenReviewer.pdf](2412.11948_OpenReviewer.pdf) | OpenReviewer: A Specialized LLM for Critical Paper Reviews | Idahl & Ahmadi | 2024 | Fine-tuned LLM for reviews |
| 8 | [2501.10326_LLM_ASPR_Survey.pdf](2501.10326_LLM_ASPR_Survey.pdf) | Large Language Models for Automated Scholarly Paper Review: A Survey | Zhuang et al. | 2025 | Comprehensive ASPR survey |
| 9 | [2503.08506_ReviewAgents.pdf](2503.08506_ReviewAgents.pdf) | ReviewAgents: Bridging Human and AI-Generated Reviews | Gao et al. | 2025 | Multi-role review framework |

## Detailed Descriptions

### Core Multi-Agent Papers

**AgentReview (2406.12708)** ⭐ Highly Relevant
- First LLM-based peer review simulation framework
- Studies reviewer biases: commitment, intention, knowledgeability
- 53,800+ generated reviews
- Key finding: 37.1% decision variation due to biases
- Code: https://github.com/Ahren09/AgentReview

**MARG (2401.04259)** ⭐ Highly Relevant
- Multi-agent review generation with specialized experts
- Leader, worker, and expert agent architecture
- 2.2x improvement in helpful comments
- Code: https://github.com/allenai/marg-reviewer

**ReviewAgents (2503.08506)**
- Chain-of-thought reasoning for structured reviews
- Multi-role framework (reviewer + AC)
- Review-CoT dataset: 142K comments
- Aligns LLM behavior with human review practices

### Specialized Models

**OpenReviewer (2412.11948)**
- Llama-OpenReviewer-8B fine-tuned on 79K reviews
- Key finding: General LLMs are overly positive
- Produces more critical, realistic reviews
- Model: huggingface.co/maxidl/Llama-OpenReviewer-8B

### Surveys

**LLM-as-a-Judge Survey (2411.15594)**
- Evaluation paradigms: pointwise, pairwise, listwise
- Bias analysis: position, self-preference, verbosity
- Multi-model judging reduces biases

**LLMs-as-Judges Survey (2412.05579)**
- Five dimensions: Functionality, Methodology, Applications, Meta-evaluation, Limitations
- Identifies preference leakage issue
- Calls for standardized evaluation

**ASPR Survey (2501.10326)**
- Comprehensive review of LLMs in automated paper review
- Technological progress and remaining challenges
- Future directions for the field

### Foundation

**PeerRead (1804.09635)**
- First public peer review dataset
- 14.7K papers, 10.7K reviews
- Venues: ACL, NIPS, ICLR
- Data: https://github.com/allenai/PeerRead

## Relevance to Research Hypothesis

The research hypothesis states:
> "Different AI models may provide distinct opinions and suggestions when reviewing papers, and the dynamics of paper improvement may change when multiple models are involved in the review process."

**Supporting Evidence from Papers:**

1. **Different models have different biases** (OpenReviewer, LLM-as-Judge surveys)
   - General LLMs: overly positive
   - Specialized models: more critical
   - Each model family has systematic tendencies

2. **Multi-agent systems produce different dynamics** (AgentReview, MARG)
   - Agent interactions affect final outcomes
   - Biased agents can influence unbiased ones
   - Group dynamics (echo chamber, authority bias)

3. **No existing work studies cross-model equilibrium**
   - Current multi-agent work uses same model (GPT-4)
   - Gap: What happens with GPT-4, Claude, Gemini together?
   - This is the novel contribution opportunity
