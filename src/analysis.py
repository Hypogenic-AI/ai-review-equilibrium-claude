"""
Analysis module for comparing AI reviews.

Provides metrics for inter-model agreement, comparison with human reviews,
and statistical analysis of review differences.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def cohens_kappa(ratings1: List[int], ratings2: List[int]) -> float:
    """
    Calculate Cohen's Kappa for two lists of ratings.

    Args:
        ratings1: First list of ratings
        ratings2: Second list of ratings

    Returns:
        Cohen's Kappa coefficient
    """
    if len(ratings1) != len(ratings2) or len(ratings1) == 0:
        return np.nan

    # Get all unique categories
    all_ratings = set(ratings1) | set(ratings2)

    # Build confusion matrix
    n = len(ratings1)
    confusion = {}
    for r1 in all_ratings:
        for r2 in all_ratings:
            confusion[(r1, r2)] = sum(
                1 for i in range(n) if ratings1[i] == r1 and ratings2[i] == r2
            )

    # Calculate observed agreement
    po = sum(confusion.get((r, r), 0) for r in all_ratings) / n

    # Calculate expected agreement
    p1 = {r: sum(1 for x in ratings1 if x == r) / n for r in all_ratings}
    p2 = {r: sum(1 for x in ratings2 if x == r) / n for r in all_ratings}
    pe = sum(p1.get(r, 0) * p2.get(r, 0) for r in all_ratings)

    # Calculate kappa
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def cohens_kappa_weighted(ratings1: List[float], ratings2: List[float]) -> float:
    """
    Calculate weighted Cohen's Kappa for continuous ratings (binned into categories).

    Args:
        ratings1: First list of ratings
        ratings2: Second list of ratings

    Returns:
        Weighted Cohen's Kappa
    """
    if len(ratings1) != len(ratings2) or len(ratings1) == 0:
        return np.nan

    # Bin ratings: 1-3 (reject), 4-6 (borderline), 7-10 (accept)
    def bin_rating(r):
        if r <= 3:
            return "reject"
        elif r <= 6:
            return "borderline"
        else:
            return "accept"

    binned1 = [bin_rating(r) for r in ratings1]
    binned2 = [bin_rating(r) for r in ratings2]

    return cohens_kappa(binned1, binned2)


def fleiss_kappa(ratings_matrix: List[List[int]]) -> float:
    """
    Calculate Fleiss' Kappa for multiple raters.

    Args:
        ratings_matrix: Matrix where each row is a subject (paper)
                       and each column is a rater's rating category count

    Returns:
        Fleiss' Kappa coefficient
    """
    if not ratings_matrix or not ratings_matrix[0]:
        return np.nan

    n = len(ratings_matrix)  # number of subjects
    k = len(ratings_matrix[0])  # number of categories
    N = sum(ratings_matrix[0])  # number of raters per subject

    # Calculate P_i for each subject
    P = []
    for row in ratings_matrix:
        n_i = sum(row)
        if n_i <= 1:
            P.append(0)
        else:
            p_i = (sum(r * r for r in row) - n_i) / (n_i * (n_i - 1))
            P.append(p_i)

    P_bar = sum(P) / n

    # Calculate p_j for each category
    p = [sum(row[j] for row in ratings_matrix) / (n * N) for j in range(k)]

    # Calculate Pe
    Pe = sum(pj * pj for pj in p)

    # Calculate kappa
    if Pe == 1:
        return 1.0
    return (P_bar - Pe) / (1 - Pe)


def rating_correlation(
    ratings1: List[float],
    ratings2: List[float],
    method: str = "spearman"
) -> Tuple[float, float]:
    """
    Calculate correlation between two sets of ratings.

    Args:
        ratings1: First list of ratings
        ratings2: Second list of ratings
        method: 'spearman' or 'pearson'

    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    if len(ratings1) != len(ratings2) or len(ratings1) < 3:
        return np.nan, np.nan

    if method == "spearman":
        return stats.spearmanr(ratings1, ratings2)
    else:
        return stats.pearsonr(ratings1, ratings2)


def rating_statistics(ratings: List[float]) -> Dict[str, float]:
    """
    Calculate descriptive statistics for a list of ratings.

    Args:
        ratings: List of ratings

    Returns:
        Dictionary with statistics
    """
    if not ratings:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "median": np.nan}

    return {
        "mean": np.mean(ratings),
        "std": np.std(ratings),
        "min": np.min(ratings),
        "max": np.max(ratings),
        "median": np.median(ratings),
        "count": len(ratings)
    }


def recommendation_agreement(recs1: List[str], recs2: List[str]) -> Dict[str, Any]:
    """
    Calculate agreement metrics for accept/reject recommendations.

    Args:
        recs1: First list of recommendations
        recs2: Second list of recommendations

    Returns:
        Dictionary with agreement metrics
    """
    if len(recs1) != len(recs2) or len(recs1) == 0:
        return {"agreement_rate": np.nan, "kappa": np.nan}

    # Normalize recommendations
    def normalize(rec):
        rec = str(rec).lower()
        if 'accept' in rec:
            return 'accept'
        elif 'reject' in rec:
            return 'reject'
        else:
            return 'borderline'

    norm1 = [normalize(r) for r in recs1]
    norm2 = [normalize(r) for r in recs2]

    # Calculate agreement rate
    agreement = sum(1 for a, b in zip(norm1, norm2) if a == b) / len(norm1)

    # Calculate kappa
    kappa = cohens_kappa(norm1, norm2)

    return {
        "agreement_rate": agreement,
        "kappa": kappa,
        "confusion": {
            (r1, r2): sum(1 for a, b in zip(norm1, norm2) if a == r1 and b == r2)
            for r1 in ['accept', 'reject', 'borderline']
            for r2 in ['accept', 'reject', 'borderline']
        }
    }


def analyze_review_content(review: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze content of a single review.

    Args:
        review: Parsed review dictionary

    Returns:
        Dictionary with content analysis
    """
    if not review:
        return {}

    return {
        "num_strengths": len(review.get("strengths", [])),
        "num_weaknesses": len(review.get("weaknesses", [])),
        "num_questions": len(review.get("questions", [])),
        "summary_length": len(str(review.get("summary", "")).split()),
        "total_strength_words": sum(len(s.split()) for s in review.get("strengths", [])),
        "total_weakness_words": sum(len(w.split()) for w in review.get("weaknesses", [])),
        "strength_weakness_ratio": (
            len(review.get("strengths", [])) / max(len(review.get("weaknesses", [])), 1)
        )
    }


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def extract_keywords(text: str, min_length: int = 4) -> set:
    """Extract keywords from text (simple word-based extraction)."""
    if not text:
        return set()
    words = text.lower().split()
    # Filter common words and short words
    stopwords = {'the', 'and', 'this', 'that', 'with', 'for', 'are', 'but', 'not',
                 'from', 'have', 'has', 'been', 'would', 'could', 'should', 'will',
                 'paper', 'work', 'method', 'approach', 'results', 'however',
                 'also', 'some', 'more', 'than', 'very', 'well', 'good', 'they'}
    return {w for w in words if len(w) >= min_length and w not in stopwords}


def compare_review_content(
    review1: Dict[str, Any],
    review2: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compare content similarity between two reviews.

    Args:
        review1: First parsed review
        review2: Second parsed review

    Returns:
        Dictionary with similarity metrics
    """
    if not review1 or not review2:
        return {"keyword_overlap": np.nan, "aspect_coverage_diff": np.nan}

    # Extract keywords from all text
    text1 = " ".join([
        str(review1.get("summary", "")),
        " ".join(review1.get("strengths", [])),
        " ".join(review1.get("weaknesses", []))
    ])
    text2 = " ".join([
        str(review2.get("summary", "")),
        " ".join(review2.get("strengths", [])),
        " ".join(review2.get("weaknesses", []))
    ])

    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)

    # Keyword overlap
    keyword_sim = jaccard_similarity(keywords1, keywords2)

    # Aspect coverage difference
    content1 = analyze_review_content(review1)
    content2 = analyze_review_content(review2)

    aspect_diff = abs(
        content1.get("strength_weakness_ratio", 1) - content2.get("strength_weakness_ratio", 1)
    )

    return {
        "keyword_overlap": keyword_sim,
        "aspect_coverage_diff": aspect_diff,
        "strength_diff": abs(content1.get("num_strengths", 0) - content2.get("num_strengths", 0)),
        "weakness_diff": abs(content1.get("num_weaknesses", 0) - content2.get("num_weaknesses", 0))
    }


def compute_all_metrics(
    results: Dict[str, Dict[str, Any]],
    human_ratings: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Compute all analysis metrics for a set of model reviews.

    Args:
        results: Dictionary mapping paper_id to {model_id: review_dict}
        human_ratings: Optional list of human ratings for comparison

    Returns:
        Comprehensive analysis results
    """
    metrics = {
        "per_model_stats": {},
        "pairwise_agreement": {},
        "pairwise_correlation": {},
        "content_analysis": {},
        "overall": {}
    }

    # Collect ratings by model
    model_ratings = {}
    model_recommendations = {}
    model_reviews = {}

    for paper_id, paper_results in results.items():
        for model_id, result in paper_results.items():
            if model_id not in model_ratings:
                model_ratings[model_id] = []
                model_recommendations[model_id] = []
                model_reviews[model_id] = []

            if result.get("success") and result.get("parsed_review"):
                review = result["parsed_review"]
                rating = review.get("rating")
                rec = review.get("recommendation")

                if rating is not None:
                    model_ratings[model_id].append(rating)
                if rec is not None:
                    model_recommendations[model_id].append(rec)
                model_reviews[model_id].append(review)

    # Per-model statistics
    for model_id, ratings in model_ratings.items():
        metrics["per_model_stats"][model_id] = rating_statistics(ratings)
        metrics["per_model_stats"][model_id]["recommendations"] = Counter(
            [normalize_rec(r) for r in model_recommendations.get(model_id, [])]
        )

    # Pairwise agreement and correlation
    models = list(model_ratings.keys())
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            # Get aligned ratings (only papers where both models have ratings)
            paired = []
            for paper_id in results.keys():
                r1 = results[paper_id].get(model1, {}).get("parsed_review", {}).get("rating")
                r2 = results[paper_id].get(model2, {}).get("parsed_review", {}).get("rating")
                if r1 is not None and r2 is not None:
                    paired.append((r1, r2))

            if paired:
                ratings1 = [p[0] for p in paired]
                ratings2 = [p[1] for p in paired]

                pair_key = f"{model1}_vs_{model2}"
                metrics["pairwise_agreement"][pair_key] = {
                    "kappa": cohens_kappa_weighted(ratings1, ratings2),
                    "n_papers": len(paired)
                }
                corr, p_val = rating_correlation(ratings1, ratings2)
                metrics["pairwise_correlation"][pair_key] = {
                    "spearman_r": corr,
                    "p_value": p_val
                }

    # Overall metrics
    all_ratings = [r for ratings in model_ratings.values() for r in ratings]
    metrics["overall"] = {
        "total_papers": len(results),
        "total_reviews": len(all_ratings),
        "mean_rating_all_models": np.mean(all_ratings) if all_ratings else np.nan,
        "std_rating_all_models": np.std(all_ratings) if all_ratings else np.nan
    }

    return metrics


def normalize_rec(rec):
    """Normalize recommendation string."""
    rec = str(rec).lower()
    if 'accept' in rec:
        return 'accept'
    elif 'reject' in rec:
        return 'reject'
    return 'borderline'


def statistical_tests(
    ratings1: List[float],
    ratings2: List[float],
    name1: str = "Model 1",
    name2: str = "Model 2"
) -> Dict[str, Any]:
    """
    Perform statistical tests comparing two sets of ratings.

    Args:
        ratings1: First list of ratings
        ratings2: Second list of ratings
        name1: Name of first model
        name2: Name of second model

    Returns:
        Dictionary with test results
    """
    if len(ratings1) < 3 or len(ratings2) < 3:
        return {"error": "Insufficient data for statistical tests"}

    results = {}

    # Paired t-test (if same length)
    if len(ratings1) == len(ratings2):
        t_stat, t_pval = stats.ttest_rel(ratings1, ratings2)
        results["paired_ttest"] = {"t_statistic": t_stat, "p_value": t_pval}

        # Wilcoxon signed-rank test
        w_stat, w_pval = stats.wilcoxon(ratings1, ratings2)
        results["wilcoxon"] = {"statistic": w_stat, "p_value": w_pval}

    # Independent t-test
    t_stat, t_pval = stats.ttest_ind(ratings1, ratings2)
    results["independent_ttest"] = {"t_statistic": t_stat, "p_value": t_pval}

    # Mann-Whitney U test
    u_stat, u_pval = stats.mannwhitneyu(ratings1, ratings2)
    results["mann_whitney"] = {"statistic": u_stat, "p_value": u_pval}

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(ratings1)-1)*np.std(ratings1)**2 + (len(ratings2)-1)*np.std(ratings2)**2) /
        (len(ratings1) + len(ratings2) - 2)
    )
    cohens_d = (np.mean(ratings1) - np.mean(ratings2)) / pooled_std if pooled_std > 0 else 0
    results["effect_size"] = {"cohens_d": cohens_d}

    return results


if __name__ == "__main__":
    # Test with synthetic data
    ratings_a = [7, 8, 6, 5, 7, 8, 6, 7, 8, 9]
    ratings_b = [6, 7, 5, 4, 6, 7, 5, 6, 7, 8]
    ratings_c = [5, 6, 4, 3, 5, 6, 4, 5, 6, 7]

    print("Testing analysis functions...")
    print(f"Kappa A vs B: {cohens_kappa_weighted(ratings_a, ratings_b):.3f}")
    print(f"Kappa A vs C: {cohens_kappa_weighted(ratings_a, ratings_c):.3f}")
    print(f"Kappa B vs C: {cohens_kappa_weighted(ratings_b, ratings_c):.3f}")

    corr, pval = rating_correlation(ratings_a, ratings_b)
    print(f"Spearman A vs B: r={corr:.3f}, p={pval:.4f}")

    tests = statistical_tests(ratings_a, ratings_b, "A", "B")
    print(f"T-test: t={tests['paired_ttest']['t_statistic']:.3f}, p={tests['paired_ttest']['p_value']:.4f}")
    print(f"Cohen's d: {tests['effect_size']['cohens_d']:.3f}")
