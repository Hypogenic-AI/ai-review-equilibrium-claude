"""
Data loader for PeerRead dataset.

Loads and preprocesses papers and reviews from the PeerRead ICLR 2017 dataset.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import random

# Set random seed for reproducibility
random.seed(42)


def load_iclr_papers(data_dir: str, split: str = "train") -> List[Dict[str, Any]]:
    """
    Load papers from the ICLR 2017 PeerRead dataset.

    Args:
        data_dir: Path to PeerRead_raw directory
        split: One of 'train', 'dev', 'test'

    Returns:
        List of paper dictionaries with reviews
    """
    reviews_path = Path(data_dir) / "data" / "iclr_2017" / split / "reviews"

    if not reviews_path.exists():
        raise FileNotFoundError(f"Reviews path not found: {reviews_path}")

    papers = []
    for filename in sorted(os.listdir(reviews_path)):
        if filename.endswith('.json'):
            with open(reviews_path / filename, 'r') as f:
                paper_data = json.load(f)
                papers.append(paper_data)

    return papers


def filter_papers_with_reviews(papers: List[Dict], min_reviews: int = 3) -> List[Dict]:
    """
    Filter papers that have at least min_reviews non-meta reviews.

    Args:
        papers: List of paper dictionaries
        min_reviews: Minimum number of reviews required

    Returns:
        Filtered list of papers
    """
    filtered = []
    for paper in papers:
        reviews = paper.get('reviews', [])
        # Count non-meta reviews with actual content
        non_meta_reviews = [
            r for r in reviews
            if not r.get('IS_META_REVIEW', False)
            and r.get('comments', '').strip()
            and 'RECOMMENDATION' in r
        ]
        if len(non_meta_reviews) >= min_reviews:
            paper['filtered_reviews'] = non_meta_reviews
            filtered.append(paper)
    return filtered


def extract_paper_info(paper: Dict) -> Dict[str, Any]:
    """
    Extract relevant information from a paper for review generation.

    Args:
        paper: Paper dictionary from PeerRead

    Returns:
        Dictionary with paper info for review prompts
    """
    return {
        'id': paper.get('id', 'unknown'),
        'title': paper.get('title', ''),
        'abstract': paper.get('abstract', ''),
        'authors': paper.get('authors', ''),
        'accepted': paper.get('accepted', None),
        'conference': paper.get('conference', 'ICLR 2017'),
        'human_reviews': paper.get('filtered_reviews', [])
    }


def extract_human_review_stats(paper: Dict) -> Dict[str, Any]:
    """
    Extract statistics from human reviews for comparison.

    Args:
        paper: Paper dictionary with filtered_reviews

    Returns:
        Dictionary with review statistics
    """
    reviews = paper.get('filtered_reviews', [])

    recommendations = [r.get('RECOMMENDATION') for r in reviews if r.get('RECOMMENDATION')]
    confidences = [r.get('REVIEWER_CONFIDENCE') for r in reviews if r.get('REVIEWER_CONFIDENCE')]
    originalities = [r.get('ORIGINALITY') for r in reviews if r.get('ORIGINALITY')]
    clarities = [r.get('CLARITY') for r in reviews if r.get('CLARITY')]

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else None

    return {
        'num_reviews': len(reviews),
        'recommendations': recommendations,
        'mean_recommendation': safe_mean(recommendations),
        'mean_confidence': safe_mean(confidences),
        'mean_originality': safe_mean(originalities),
        'mean_clarity': safe_mean(clarities),
        'review_texts': [r.get('comments', '') for r in reviews]
    }


def select_papers_for_experiment(
    papers: List[Dict],
    n_accepted: int = 10,
    n_rejected: int = 5,
    abstract_min_words: int = 100,
    abstract_max_words: int = 500
) -> List[Dict]:
    """
    Select a balanced set of papers for the experiment.

    Args:
        papers: List of filtered papers
        n_accepted: Number of accepted papers to select
        n_rejected: Number of rejected papers to select
        abstract_min_words: Minimum words in abstract
        abstract_max_words: Maximum words in abstract

    Returns:
        Selected list of papers
    """
    # Filter by abstract length
    valid_papers = []
    for paper in papers:
        abstract = paper.get('abstract', '')
        word_count = len(abstract.split())
        if abstract_min_words <= word_count <= abstract_max_words:
            valid_papers.append(paper)

    # Separate by acceptance
    accepted = [p for p in valid_papers if p.get('accepted', False)]
    rejected = [p for p in valid_papers if not p.get('accepted', False)]

    # Sample
    random.shuffle(accepted)
    random.shuffle(rejected)

    selected_accepted = accepted[:n_accepted]
    selected_rejected = rejected[:n_rejected]

    return selected_accepted + selected_rejected


def prepare_experiment_dataset(peerread_path: str) -> List[Dict]:
    """
    Prepare the full experiment dataset.

    Args:
        peerread_path: Path to PeerRead_raw directory

    Returns:
        List of prepared paper dictionaries
    """
    # Load all papers from train and test splits
    all_papers = []
    for split in ['train', 'dev', 'test']:
        try:
            papers = load_iclr_papers(peerread_path, split)
            all_papers.extend(papers)
            print(f"Loaded {len(papers)} papers from {split} split")
        except FileNotFoundError:
            print(f"Warning: {split} split not found")

    print(f"Total papers loaded: {len(all_papers)}")

    # Filter papers with sufficient reviews
    filtered = filter_papers_with_reviews(all_papers, min_reviews=2)
    print(f"Papers with 2+ reviews: {len(filtered)}")

    # Select balanced set
    selected = select_papers_for_experiment(filtered, n_accepted=10, n_rejected=5)
    print(f"Selected papers: {len(selected)} (accepted: {sum(1 for p in selected if p.get('accepted'))}, rejected: {sum(1 for p in selected if not p.get('accepted'))})")

    # Extract info and stats
    experiment_papers = []
    for paper in selected:
        info = extract_paper_info(paper)
        stats = extract_human_review_stats(paper)
        experiment_papers.append({**info, 'human_stats': stats})

    return experiment_papers


if __name__ == "__main__":
    # Test the data loader
    peerread_path = "/data/hypogenicai/workspaces/ai-review-equilibrium-claude/datasets/PeerRead_raw"
    papers = prepare_experiment_dataset(peerread_path)

    print("\n--- Sample Paper ---")
    sample = papers[0]
    print(f"Title: {sample['title']}")
    print(f"Abstract length: {len(sample['abstract'].split())} words")
    print(f"Accepted: {sample['accepted']}")
    print(f"Human reviews: {sample['human_stats']['num_reviews']}")
    print(f"Mean recommendation: {sample['human_stats']['mean_recommendation']:.2f}")
