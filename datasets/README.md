# Downloaded Datasets

This directory contains datasets for the AI Reviewing Equilibrium research project.
Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: PeerRead

### Overview
- **Source**: https://github.com/allenai/PeerRead
- **Size**: 14.7K paper drafts, 10.7K reviews
- **Format**: JSON files with papers and reviews
- **Task**: Peer review analysis, accept/reject prediction
- **Splits**: train, dev, test for each venue
- **License**: Academic use

### Dataset Contents
| Venue | Papers | Reviews | Notes |
|-------|--------|---------|-------|
| ICLR 2017 | 427 | 1,304 | Full reviews with scores |
| ACL 2017 | 137 | 275 | Reviews with aspect scores |
| CoNLL 2016 | 22 | 39 | Reviews with aspect scores |
| NIPS 2013-2017 | 2,420 | 9,152 | Accepted papers only |
| arXiv 2007-2017 | 11,778 | — | Accept/reject labels only |

### Download Instructions

**Using Git (recommended):**
```bash
git clone https://github.com/allenai/PeerRead.git datasets/PeerRead_raw --depth 1
```

**Current Location:**
`datasets/PeerRead_raw/`

### Loading the Dataset

```python
import json
import os

# Load ICLR 2017 reviews
iclr_path = "datasets/PeerRead_raw/data/iclr_2017/train/reviews"
for filename in os.listdir(iclr_path):
    with open(os.path.join(iclr_path, filename)) as f:
        paper_data = json.load(f)
        print(f"Paper: {paper_data['title']}")
        print(f"Reviews: {len(paper_data['reviews'])}")
        print(f"Accepted: {paper_data['accepted']}")
```

### Sample Data

Example record structure:
```json
{
    "conference": "ICLR 2017 conference submission",
    "title": "Paper Title",
    "abstract": "Paper abstract...",
    "authors": "Author1, Author2",
    "accepted": true,
    "reviews": [
        {
            "IS_META_REVIEW": false,
            "TITLE": "Review Title",
            "comments": "Review text...",
            "RECOMMENDATION": 8,
            "REVIEWER_CONFIDENCE": 4,
            "ORIGINALITY": 4,
            "CLARITY": 3
        }
    ]
}
```

### Notes
- Contains both human expert reviews and paper text
- Review scores range from 1-10 for recommendation
- Aspect scores (originality, clarity, etc.) on 1-5 scale
- Can be used to study review dynamics with multiple reviewers per paper

---

## Dataset 2: SEA OpenReview Data

### Overview
- **Source**: https://huggingface.co/datasets/ECNU-SEA/SEA_data
- **Size**: 8,119 train + 902 test samples
- **Format**: HuggingFace Dataset (text format with paper IDs)
- **Task**: Paper analysis, review research

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset
dataset = load_dataset("ECNU-SEA/SEA_data")
dataset.save_to_disk("datasets/sea_openreview")
```

**Current Location:**
`datasets/sea_openreview/`

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/sea_openreview")
print(dataset)
```

---

## Dataset 3: AgentReview Simulation Data

### Overview
- **Source**: https://github.com/Ahren09/AgentReview
- **Size**: 53,800+ generated reviews, rebuttals, and meta-reviews
- **Format**: JSON/Structured text
- **Task**: Peer review simulation analysis
- **Content**: LLM-generated peer reviews simulating various reviewer biases

### Download Instructions

The AgentReview data requires downloading from Dropbox links provided in their repository.

**Step 1: Clone the repository**
```bash
git clone https://github.com/Ahren09/AgentReview.git code/AgentReview
```

**Step 2: Download data from Dropbox**
- Download `AgentReview_Paper_Data.zip` and unzip under `data/`
- Download `AgentReview_LLM_Reviews.zip` and unzip under `outputs/`

See the repository README for specific Dropbox links.

### Notes
- Contains simulated peer reviews from LLM agents
- Explores reviewer biases: commitment, intention, knowledgeability
- Useful for studying multi-agent review dynamics

---

## Dataset 4: OpenReview API Data (For Fresh Data Collection)

### Overview
For the most up-to-date peer review data, use the OpenReview API directly.

### Data Collection Script

```python
import openreview

# Connect to OpenReview API
client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

# Get ICLR 2024 submissions
submissions = client.get_all_notes(
    content={'venueid': 'ICLR.cc/2024/Conference'}
)

# Get reviews for a specific paper
paper_id = submissions[0].id
reviews = client.get_all_notes(forum=paper_id, signature='ICLR.cc/2024/Conference/Paper.*AnonReviewer.*')
```

### Notes
- Requires `openreview-py` package: `pip install openreview-py`
- Rate limits apply
- Check OpenReview terms of service for usage restrictions

---

## Recommendations for Experiments

### Primary Dataset: PeerRead (ICLR 2017)
- **Why**: Rich review data with multiple reviewers per paper, aspect scores, and accept/reject decisions
- **Use for**: Multi-model review comparison, review quality assessment

### Supplementary: AgentReview
- **Why**: Large-scale simulated data exploring reviewer biases
- **Use for**: Studying multi-agent review dynamics, bias effects

### Data Collection: OpenReview API
- **Why**: Most current data from recent conferences
- **Use for**: Generating fresh evaluation datasets

---

## Data Statistics Summary

| Dataset | Papers | Reviews | Years | Format |
|---------|--------|---------|-------|--------|
| PeerRead | 14,784 | 10,770 | 2007-2017 | JSON |
| AgentReview | 500+ | 53,800+ | Simulated | JSON |
| OpenReview API | Variable | Variable | 2017-present | API |
