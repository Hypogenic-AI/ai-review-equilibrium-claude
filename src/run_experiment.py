"""
Main experiment runner for AI Reviewing Equilibrium research.

This script orchestrates:
1. Data loading and preparation
2. Multi-model review generation
3. Analysis and comparison
4. Dynamics experiment
5. Visualization and reporting
"""

import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import prepare_experiment_dataset, extract_human_review_stats
from review_generator import (
    generate_all_reviews, result_to_dict, MODELS,
    generate_review, REVIEW_PROMPT
)
from analysis import (
    compute_all_metrics, rating_statistics, cohens_kappa_weighted,
    rating_correlation, statistical_tests, recommendation_agreement,
    analyze_review_content, compare_review_content
)

# Set random seed
random.seed(42)
np.random.seed(42)

# Paths
WORKSPACE_DIR = Path("/data/hypogenicai/workspaces/ai-review-equilibrium-claude")
PEERREAD_PATH = WORKSPACE_DIR / "datasets" / "PeerRead_raw"
RESULTS_DIR = WORKSPACE_DIR / "results"
FIGURES_DIR = WORKSPACE_DIR / "figures"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def save_results(data: Any, filename: str) -> None:
    """Save results to JSON file."""
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {filepath}")


def load_results(filename: str) -> Any:
    """Load results from JSON file."""
    filepath = RESULTS_DIR / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def experiment_1_multimodel_comparison(
    papers: List[Dict],
    models: List[str] = ["gpt4", "claude", "gemini"],
    save_intermediate: bool = True
) -> Dict[str, Any]:
    """
    Experiment 1: Generate and compare reviews from multiple AI models.

    Args:
        papers: List of paper dictionaries
        models: List of model IDs
        save_intermediate: Whether to save results after each paper

    Returns:
        Dictionary with all results
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Multi-Model Review Comparison")
    print("="*60)

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_papers": len(papers),
            "models": models,
            "model_names": {m: MODELS[m]["display_name"] for m in models}
        },
        "papers": {},
        "raw_reviews": {}
    }

    for i, paper in enumerate(papers):
        print(f"\n[{i+1}/{len(papers)}] Reviewing: {paper['title'][:60]}...")

        paper_id = paper['id']
        results["papers"][paper_id] = {
            "title": paper['title'],
            "abstract": paper['abstract'][:500] + "..." if len(paper['abstract']) > 500 else paper['abstract'],
            "accepted": paper['accepted'],
            "human_stats": paper.get('human_stats', {})
        }

        # Generate reviews from all models
        model_results = generate_all_reviews(paper, models=models, delay_between_calls=0.5)

        # Convert to serializable format
        results["raw_reviews"][paper_id] = {
            model_id: result_to_dict(result)
            for model_id, result in model_results.items()
        }

        if save_intermediate:
            save_results(results, "experiment_1_intermediate.json")

    print("\n--- Review Generation Complete ---")

    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_multimodel_results(results)
    results["analysis"] = analysis

    save_results(results, "experiment_1_results.json")
    return results


def analyze_multimodel_results(results: Dict) -> Dict[str, Any]:
    """
    Analyze multi-model review results.

    Args:
        results: Raw experiment results

    Returns:
        Analysis dictionary
    """
    analysis = {
        "per_model_stats": {},
        "pairwise_comparison": {},
        "vs_human": {},
        "content_analysis": {}
    }

    models = results["metadata"]["models"]
    raw_reviews = results["raw_reviews"]

    # Collect ratings by model
    model_ratings = {m: [] for m in models}
    model_recs = {m: [] for m in models}
    model_parsed = {m: [] for m in models}

    paper_ratings = {}  # paper_id -> {model: rating}

    for paper_id, paper_reviews in raw_reviews.items():
        paper_ratings[paper_id] = {}

        for model_id, review_data in paper_reviews.items():
            if review_data.get("success") and review_data.get("parsed_review"):
                parsed = review_data["parsed_review"]
                rating = parsed.get("rating")
                rec = parsed.get("recommendation")

                if rating is not None:
                    model_ratings[model_id].append(rating)
                    paper_ratings[paper_id][model_id] = rating
                if rec is not None:
                    model_recs[model_id].append(rec)
                model_parsed[model_id].append(parsed)

    # Per-model statistics
    for model_id in models:
        ratings = model_ratings[model_id]
        recs = model_recs[model_id]

        analysis["per_model_stats"][model_id] = {
            **rating_statistics(ratings),
            "recommendation_dist": {
                "accept": sum(1 for r in recs if 'accept' in str(r).lower() and 'reject' not in str(r).lower()),
                "reject": sum(1 for r in recs if 'reject' in str(r).lower()),
                "borderline": sum(1 for r in recs if 'borderline' in str(r).lower())
            },
            "success_rate": len(ratings) / len(raw_reviews) if raw_reviews else 0
        }

    # Pairwise comparisons
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            # Get aligned ratings
            ratings1, ratings2 = [], []
            for paper_id in paper_ratings.keys():
                r1 = paper_ratings[paper_id].get(model1)
                r2 = paper_ratings[paper_id].get(model2)
                if r1 is not None and r2 is not None:
                    ratings1.append(r1)
                    ratings2.append(r2)

            if len(ratings1) >= 3:
                pair_key = f"{model1}_vs_{model2}"

                kappa = cohens_kappa_weighted(ratings1, ratings2)
                corr, pval = rating_correlation(ratings1, ratings2)
                tests = statistical_tests(ratings1, ratings2, model1, model2)

                analysis["pairwise_comparison"][pair_key] = {
                    "n_papers": len(ratings1),
                    "kappa": kappa,
                    "spearman_r": corr,
                    "spearman_p": pval,
                    "mean_difference": np.mean(ratings1) - np.mean(ratings2),
                    "cohens_d": tests["effect_size"]["cohens_d"],
                    "paired_ttest_p": tests.get("paired_ttest", {}).get("p_value"),
                    "wilcoxon_p": tests.get("wilcoxon", {}).get("p_value")
                }

    # Comparison vs human ratings
    for paper_id, paper_data in results["papers"].items():
        human_stats = paper_data.get("human_stats", {})
        human_mean = human_stats.get("mean_recommendation")

        if human_mean is not None:
            for model_id in models:
                model_rating = paper_ratings[paper_id].get(model_id)
                if model_rating is not None:
                    if model_id not in analysis["vs_human"]:
                        analysis["vs_human"][model_id] = {"diffs": [], "human_ratings": [], "model_ratings": []}
                    analysis["vs_human"][model_id]["diffs"].append(model_rating - human_mean)
                    analysis["vs_human"][model_id]["human_ratings"].append(human_mean)
                    analysis["vs_human"][model_id]["model_ratings"].append(model_rating)

    # Compute human comparison stats
    for model_id, data in analysis["vs_human"].items():
        if len(data["diffs"]) >= 3:
            corr, pval = rating_correlation(data["human_ratings"], data["model_ratings"])
            analysis["vs_human"][model_id]["stats"] = {
                "mean_diff": np.mean(data["diffs"]),
                "std_diff": np.std(data["diffs"]),
                "correlation": corr,
                "correlation_p": pval,
                "n_papers": len(data["diffs"])
            }

    # Content analysis
    for model_id in models:
        parsed_reviews = model_parsed[model_id]
        if parsed_reviews:
            content_stats = [analyze_review_content(r) for r in parsed_reviews]
            analysis["content_analysis"][model_id] = {
                "avg_strengths": np.mean([c["num_strengths"] for c in content_stats]),
                "avg_weaknesses": np.mean([c["num_weaknesses"] for c in content_stats]),
                "avg_questions": np.mean([c["num_questions"] for c in content_stats]),
                "avg_strength_words": np.mean([c["total_strength_words"] for c in content_stats]),
                "avg_weakness_words": np.mean([c["total_weakness_words"] for c in content_stats]),
                "avg_sw_ratio": np.mean([c["strength_weakness_ratio"] for c in content_stats])
            }

    return analysis


def experiment_2_dynamics(
    papers: List[Dict],
    initial_results: Dict,
    n_iterations: int = 2
) -> Dict[str, Any]:
    """
    Experiment 2: Paper improvement dynamics.

    Simulate paper revision based on multi-model feedback and track opinion changes.

    Args:
        papers: List of paper dictionaries
        initial_results: Results from experiment 1
        n_iterations: Number of revision iterations

    Returns:
        Dynamics experiment results
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Paper Improvement Dynamics")
    print("="*60)

    # Select subset of papers for dynamics experiment
    subset_papers = papers[:5]  # Use first 5 papers

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_papers": len(subset_papers),
            "n_iterations": n_iterations
        },
        "dynamics": {}
    }

    for paper in subset_papers:
        paper_id = paper['id']
        print(f"\n[Dynamics] Paper: {paper['title'][:50]}...")

        paper_dynamics = {
            "iterations": [],
            "initial_abstract": paper['abstract']
        }

        current_abstract = paper['abstract']
        prev_ratings = {}

        # Get initial ratings
        if paper_id in initial_results.get("raw_reviews", {}):
            for model_id, review_data in initial_results["raw_reviews"][paper_id].items():
                if review_data.get("success"):
                    rating = review_data["parsed_review"].get("rating")
                    prev_ratings[model_id] = rating

        paper_dynamics["iterations"].append({
            "iteration": 0,
            "abstract": current_abstract[:300] + "...",
            "ratings": prev_ratings.copy()
        })

        # Simulate revisions
        for iteration in range(1, n_iterations + 1):
            print(f"  Iteration {iteration}...")

            # Generate revision suggestions (simplified: use one model to suggest improvements)
            revision_prompt = f"""Based on this paper abstract:

{current_abstract}

Suggest a brief improvement (2-3 sentences) to address common reviewer concerns about:
1. Clarity of the main contribution
2. Experimental rigor

Output ONLY the improved abstract paragraph (no explanations):"""

            from review_generator import create_openai_client
            client = create_openai_client()

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": revision_prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                revised_abstract = response.choices[0].message.content.strip()

                # Get new reviews for revised abstract
                revised_paper = {"id": paper_id, "title": paper['title'], "abstract": revised_abstract}
                new_reviews = generate_all_reviews(revised_paper, models=["gpt4", "claude", "gemini"], delay_between_calls=0.3)

                new_ratings = {}
                for model_id, result in new_reviews.items():
                    if result.success and result.parsed_review:
                        new_ratings[model_id] = result.parsed_review.get("rating")

                paper_dynamics["iterations"].append({
                    "iteration": iteration,
                    "abstract": revised_abstract[:300] + "...",
                    "ratings": new_ratings,
                    "rating_changes": {
                        model_id: new_ratings.get(model_id, 0) - prev_ratings.get(model_id, 0)
                        for model_id in new_ratings
                    }
                })

                prev_ratings = new_ratings.copy()
                current_abstract = revised_abstract

            except Exception as e:
                print(f"    Error in iteration: {e}")
                break

        results["dynamics"][paper_id] = paper_dynamics

    # Analyze dynamics
    results["analysis"] = analyze_dynamics(results)

    save_results(results, "experiment_2_dynamics.json")
    return results


def analyze_dynamics(results: Dict) -> Dict[str, Any]:
    """
    Analyze dynamics results for convergence/divergence patterns.

    Args:
        results: Dynamics experiment results

    Returns:
        Analysis dictionary
    """
    analysis = {
        "convergence_metrics": {},
        "per_model_trends": {},
        "overall_patterns": {}
    }

    all_changes = {"gpt4": [], "claude": [], "gemini": []}

    for paper_id, paper_dynamics in results.get("dynamics", {}).items():
        iterations = paper_dynamics.get("iterations", [])

        if len(iterations) > 1:
            # Track rating changes across iterations
            for it in iterations[1:]:
                for model_id, change in it.get("rating_changes", {}).items():
                    if model_id in all_changes and change is not None:
                        all_changes[model_id].append(change)

            # Check convergence (do models agree more over iterations?)
            initial_ratings = list(iterations[0].get("ratings", {}).values())
            final_ratings = list(iterations[-1].get("ratings", {}).values())

            if len(initial_ratings) >= 2 and len(final_ratings) >= 2:
                initial_std = np.std([r for r in initial_ratings if r is not None])
                final_std = np.std([r for r in final_ratings if r is not None])

                analysis["convergence_metrics"][paper_id] = {
                    "initial_std": initial_std,
                    "final_std": final_std,
                    "converged": final_std < initial_std,
                    "std_change": final_std - initial_std
                }

    # Per-model trends
    for model_id, changes in all_changes.items():
        if changes:
            analysis["per_model_trends"][model_id] = {
                "mean_change": np.mean(changes),
                "positive_changes": sum(1 for c in changes if c > 0),
                "negative_changes": sum(1 for c in changes if c < 0),
                "no_change": sum(1 for c in changes if c == 0)
            }

    # Overall patterns
    all_stds = [m.get("std_change", 0) for m in analysis["convergence_metrics"].values()]
    if all_stds:
        analysis["overall_patterns"] = {
            "mean_std_change": np.mean(all_stds),
            "papers_converged": sum(1 for m in analysis["convergence_metrics"].values() if m.get("converged")),
            "papers_diverged": sum(1 for m in analysis["convergence_metrics"].values() if not m.get("converged")),
            "total_papers": len(analysis["convergence_metrics"])
        }

    return analysis


def create_visualizations(exp1_results: Dict, exp2_results: Dict = None) -> None:
    """
    Create visualizations for the experiment results.

    Args:
        exp1_results: Results from experiment 1
        exp2_results: Results from experiment 2 (optional)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use('seaborn-v0_8-whitegrid')

    analysis = exp1_results.get("analysis", {})

    # Figure 1: Rating distributions by model
    fig, ax = plt.subplots(figsize=(10, 6))

    model_data = []
    model_labels = []
    for model_id, stats in analysis.get("per_model_stats", {}).items():
        model_name = exp1_results["metadata"]["model_names"].get(model_id, model_id)
        # Get actual ratings from raw reviews
        ratings = []
        for paper_reviews in exp1_results.get("raw_reviews", {}).values():
            review_data = paper_reviews.get(model_id, {})
            if review_data.get("success"):
                rating = review_data["parsed_review"].get("rating")
                if rating is not None:
                    ratings.append(rating)
        if ratings:
            model_data.append(ratings)
            model_labels.append(model_name)

    if model_data:
        positions = list(range(len(model_labels)))
        bp = ax.boxplot(model_data, positions=positions, widths=0.6, patch_artist=True)

        colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(model_labels)]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(model_labels, fontsize=12)
        ax.set_ylabel('Rating (1-10)', fontsize=12)
        ax.set_title('Rating Distribution by AI Model', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 11)
        ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Borderline')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'rating_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {FIGURES_DIR / 'rating_distributions.png'}")

    # Figure 2: Pairwise agreement heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    pairwise = analysis.get("pairwise_comparison", {})
    models = list(analysis.get("per_model_stats", {}).keys())
    model_names = [exp1_results["metadata"]["model_names"].get(m, m) for m in models]

    n_models = len(models)
    agreement_matrix = np.ones((n_models, n_models))
    correlation_matrix = np.ones((n_models, n_models))

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i < j:
                pair_key = f"{m1}_vs_{m2}"
                if pair_key in pairwise:
                    corr = pairwise[pair_key].get("spearman_r", np.nan)
                    agreement_matrix[i, j] = corr
                    agreement_matrix[j, i] = corr

    if n_models > 1:
        sns.heatmap(agreement_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                    xticklabels=model_names, yticklabels=model_names,
                    vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Spearman r'})
        ax.set_title('Pairwise Rating Correlation Between Models', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'pairwise_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {FIGURES_DIR / 'pairwise_correlation.png'}")

    # Figure 3: Recommendation distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    rec_data = []
    for model_id in models:
        stats = analysis.get("per_model_stats", {}).get(model_id, {})
        rec_dist = stats.get("recommendation_dist", {})
        rec_data.append([
            rec_dist.get("accept", 0),
            rec_dist.get("borderline", 0),
            rec_dist.get("reject", 0)
        ])

    rec_data = np.array(rec_data).T
    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, rec_data[0], width, label='Accept', color='#2ecc71')
    ax.bar(x, rec_data[1], width, label='Borderline', color='#f1c40f')
    ax.bar(x + width, rec_data[2], width, label='Reject', color='#e74c3c')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Number of Papers', fontsize=12)
    ax.set_title('Recommendation Distribution by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'recommendation_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'recommendation_distribution.png'}")

    # Figure 4: Content analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    content = analysis.get("content_analysis", {})
    if content:
        metrics = ['avg_strengths', 'avg_weaknesses', 'avg_questions']
        metric_names = ['Avg Strengths', 'Avg Weaknesses', 'Avg Questions']

        x = np.arange(len(metric_names))
        width = 0.25
        colors = ['#3498db', '#e74c3c', '#2ecc71']

        for i, model_id in enumerate(models):
            model_content = content.get(model_id, {})
            values = [model_content.get(m, 0) for m in metrics]
            ax.bar(x + i*width - width, values, width,
                   label=model_names[i], color=colors[i % len(colors)])

        ax.set_xlabel('Review Component', fontsize=12)
        ax.set_ylabel('Average Count', fontsize=12)
        ax.set_title('Review Content Analysis by Model', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.legend()

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'content_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {FIGURES_DIR / 'content_analysis.png'}")

    # Figure 5: Model vs Human comparison
    vs_human = analysis.get("vs_human", {})
    if vs_human:
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, (model_id, data) in enumerate(vs_human.items()):
            if "human_ratings" in data and "model_ratings" in data:
                ax.scatter(data["human_ratings"], data["model_ratings"],
                          label=model_names[i] if i < len(model_names) else model_id,
                          alpha=0.7, s=100)

        # Add diagonal line
        ax.plot([0, 10], [0, 10], 'k--', alpha=0.5, label='Perfect Agreement')
        ax.set_xlabel('Human Rating', fontsize=12)
        ax.set_ylabel('AI Model Rating', fontsize=12)
        ax.set_title('AI Model Ratings vs Human Ratings', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.legend()
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'human_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {FIGURES_DIR / 'human_comparison.png'}")

    # Figure 6: Dynamics (if available)
    if exp2_results and exp2_results.get("dynamics"):
        fig, ax = plt.subplots(figsize=(12, 6))

        for paper_id, paper_dynamics in exp2_results["dynamics"].items():
            iterations = paper_dynamics.get("iterations", [])
            if len(iterations) > 1:
                for model_id in ["gpt4", "claude", "gemini"]:
                    ratings = []
                    iters = []
                    for it in iterations:
                        rating = it.get("ratings", {}).get(model_id)
                        if rating is not None:
                            ratings.append(rating)
                            iters.append(it["iteration"])
                    if ratings:
                        label = f"{paper_id[:8]}-{model_id}"
                        ax.plot(iters, ratings, marker='o', label=label, alpha=0.6)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Rating', fontsize=12)
        ax.set_title('Rating Evolution Across Revision Iterations', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'dynamics_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {FIGURES_DIR / 'dynamics_evolution.png'}")


def main():
    """Main experiment runner."""
    print("="*70)
    print("AI Reviewing Equilibrium - Research Experiment")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")

    # Step 1: Load and prepare data
    print("\n--- Step 1: Loading Data ---")
    papers = prepare_experiment_dataset(str(PEERREAD_PATH))

    if not papers:
        print("ERROR: No papers loaded. Check dataset path.")
        return

    print(f"Loaded {len(papers)} papers for experiment")

    # Save experiment configuration
    config = {
        "timestamp": datetime.now().isoformat(),
        "n_papers": len(papers),
        "models": list(MODELS.keys()),
        "random_seed": 42,
        "python_version": sys.version,
    }
    save_results(config, "experiment_config.json")

    # Step 2: Run Experiment 1
    print("\n--- Step 2: Running Experiment 1 ---")
    exp1_results = experiment_1_multimodel_comparison(papers, models=["gpt4", "claude", "gemini"])

    # Step 3: Run Experiment 2 (dynamics)
    print("\n--- Step 3: Running Experiment 2 ---")
    exp2_results = experiment_2_dynamics(papers, exp1_results, n_iterations=2)

    # Step 4: Create visualizations
    print("\n--- Step 4: Creating Visualizations ---")
    create_visualizations(exp1_results, exp2_results)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)

    # Print summary
    print("\n--- Summary ---")
    analysis = exp1_results.get("analysis", {})

    print("\nPer-Model Statistics:")
    for model_id, stats in analysis.get("per_model_stats", {}).items():
        print(f"  {model_id}: mean={stats.get('mean', 'N/A'):.2f}, "
              f"std={stats.get('std', 'N/A'):.2f}, n={stats.get('count', 0)}")

    print("\nPairwise Comparisons:")
    for pair, metrics in analysis.get("pairwise_comparison", {}).items():
        print(f"  {pair}: r={metrics.get('spearman_r', 'N/A'):.3f}, "
              f"kappa={metrics.get('kappa', 'N/A'):.3f}")

    if exp2_results:
        dynamics_analysis = exp2_results.get("analysis", {})
        overall = dynamics_analysis.get("overall_patterns", {})
        print(f"\nDynamics Summary:")
        print(f"  Papers converged: {overall.get('papers_converged', 'N/A')}/{overall.get('total_papers', 'N/A')}")
        print(f"  Mean std change: {overall.get('mean_std_change', 'N/A'):.3f}")

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
