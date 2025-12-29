"""
Multi-model review generation module.

Generates paper reviews using multiple LLM providers (OpenAI, Claude via OpenRouter, Gemini via OpenRouter).
"""

import os
import json
import time
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import httpx
from openai import OpenAI

# API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Model identifiers
MODELS = {
    "gpt4": {
        "provider": "openai",
        "model": "gpt-4o-mini",  # Use gpt-4o-mini for cost efficiency
        "display_name": "GPT-4o-mini"
    },
    "claude": {
        "provider": "openrouter",
        "model": "anthropic/claude-3.5-sonnet",
        "display_name": "Claude 3.5 Sonnet"
    },
    "gemini": {
        "provider": "openrouter",
        "model": "google/gemini-2.0-flash-001",
        "display_name": "Gemini 2.0 Flash"
    }
}

# Review prompt template
REVIEW_PROMPT = """You are an expert peer reviewer for a top-tier machine learning conference (ICLR).

Please review the following paper submission. Provide a thorough, fair, and constructive review.

## Paper Title:
{title}

## Paper Abstract:
{abstract}

## Your Review:

Please provide your review in the following JSON format (output ONLY valid JSON, no other text):

{{
    "summary": "A brief 2-3 sentence summary of the paper's main contribution",
    "strengths": ["List 3-5 key strengths of the paper"],
    "weaknesses": ["List 3-5 key weaknesses or areas for improvement"],
    "questions": ["List 2-3 questions for the authors"],
    "rating": <integer from 1-10 where 1=strong reject, 5=borderline, 10=strong accept>,
    "confidence": <integer from 1-5 where 1=low confidence, 5=high confidence>,
    "recommendation": "<accept/reject/borderline>"
}}

Be specific and constructive in your feedback. Consider:
- Technical soundness and rigor
- Novelty and significance of the contribution
- Clarity and quality of presentation
- Experimental validation (based on what's described in the abstract)
- Potential impact on the field
"""


@dataclass
class ReviewResult:
    """Container for review results."""
    model_id: str
    model_name: str
    paper_id: str
    raw_response: str
    parsed_review: Optional[Dict[str, Any]]
    success: bool
    error_message: Optional[str]
    latency_seconds: float


def create_openai_client() -> OpenAI:
    """Create OpenAI client."""
    return OpenAI(api_key=OPENAI_API_KEY)


def create_openrouter_client() -> OpenAI:
    """Create OpenRouter client (uses OpenAI-compatible API)."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )


def parse_review_json(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON review from model response.

    Args:
        response_text: Raw model response

    Returns:
        Parsed review dictionary or None if parsing fails
    """
    # Try to extract JSON from the response
    text = response_text.strip()

    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown code blocks
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{[\s\S]*\}'
    ]

    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue

    return None


def generate_review_openai(
    client: OpenAI,
    model: str,
    paper_title: str,
    paper_abstract: str
) -> tuple[str, float]:
    """
    Generate review using OpenAI API.

    Args:
        client: OpenAI client
        model: Model identifier
        paper_title: Paper title
        paper_abstract: Paper abstract

    Returns:
        Tuple of (response_text, latency_seconds)
    """
    prompt = REVIEW_PROMPT.format(title=paper_title, abstract=paper_abstract)

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # For reproducibility
        max_tokens=2000
    )
    latency = time.time() - start_time

    return response.choices[0].message.content, latency


def generate_review_openrouter(
    client: OpenAI,
    model: str,
    paper_title: str,
    paper_abstract: str
) -> tuple[str, float]:
    """
    Generate review using OpenRouter API.

    Args:
        client: OpenRouter client
        model: Model identifier
        paper_title: Paper title
        paper_abstract: Paper abstract

    Returns:
        Tuple of (response_text, latency_seconds)
    """
    prompt = REVIEW_PROMPT.format(title=paper_title, abstract=paper_abstract)

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2000
    )
    latency = time.time() - start_time

    return response.choices[0].message.content, latency


def generate_review(
    model_id: str,
    paper_id: str,
    paper_title: str,
    paper_abstract: str
) -> ReviewResult:
    """
    Generate a review for a paper using the specified model.

    Args:
        model_id: Model identifier (gpt4, claude, gemini)
        paper_id: Paper ID
        paper_title: Paper title
        paper_abstract: Paper abstract

    Returns:
        ReviewResult object
    """
    model_config = MODELS.get(model_id)
    if not model_config:
        return ReviewResult(
            model_id=model_id,
            model_name="Unknown",
            paper_id=paper_id,
            raw_response="",
            parsed_review=None,
            success=False,
            error_message=f"Unknown model: {model_id}",
            latency_seconds=0
        )

    try:
        if model_config["provider"] == "openai":
            client = create_openai_client()
            response_text, latency = generate_review_openai(
                client, model_config["model"], paper_title, paper_abstract
            )
        else:  # openrouter
            client = create_openrouter_client()
            response_text, latency = generate_review_openrouter(
                client, model_config["model"], paper_title, paper_abstract
            )

        parsed_review = parse_review_json(response_text)

        return ReviewResult(
            model_id=model_id,
            model_name=model_config["display_name"],
            paper_id=paper_id,
            raw_response=response_text,
            parsed_review=parsed_review,
            success=parsed_review is not None,
            error_message=None if parsed_review else "Failed to parse JSON response",
            latency_seconds=latency
        )

    except Exception as e:
        return ReviewResult(
            model_id=model_id,
            model_name=model_config["display_name"],
            paper_id=paper_id,
            raw_response="",
            parsed_review=None,
            success=False,
            error_message=str(e),
            latency_seconds=0
        )


def generate_all_reviews(
    paper: Dict[str, Any],
    models: List[str] = ["gpt4", "claude", "gemini"],
    delay_between_calls: float = 1.0
) -> Dict[str, ReviewResult]:
    """
    Generate reviews from all specified models for a paper.

    Args:
        paper: Paper dictionary with 'id', 'title', 'abstract'
        models: List of model IDs to use
        delay_between_calls: Delay in seconds between API calls

    Returns:
        Dictionary mapping model_id to ReviewResult
    """
    results = {}

    for model_id in models:
        print(f"  Generating review with {model_id}...", end=" ", flush=True)
        result = generate_review(
            model_id=model_id,
            paper_id=paper['id'],
            paper_title=paper['title'],
            paper_abstract=paper['abstract']
        )
        results[model_id] = result
        status = "OK" if result.success else f"FAILED: {result.error_message}"
        print(f"{status} ({result.latency_seconds:.1f}s)")

        if delay_between_calls > 0:
            time.sleep(delay_between_calls)

    return results


def result_to_dict(result: ReviewResult) -> Dict[str, Any]:
    """Convert ReviewResult to dictionary for JSON serialization."""
    return {
        "model_id": result.model_id,
        "model_name": result.model_name,
        "paper_id": result.paper_id,
        "raw_response": result.raw_response,
        "parsed_review": result.parsed_review,
        "success": result.success,
        "error_message": result.error_message,
        "latency_seconds": result.latency_seconds
    }


if __name__ == "__main__":
    # Test review generation
    test_paper = {
        "id": "test_001",
        "title": "A Novel Approach to Neural Network Optimization",
        "abstract": """
        We propose a new optimization algorithm for training deep neural networks
        that combines adaptive learning rates with momentum-based updates. Our method
        achieves faster convergence than standard SGD while maintaining comparable
        generalization performance. We evaluate our approach on image classification
        tasks using CIFAR-10 and ImageNet, demonstrating consistent improvements
        in training speed without sacrificing accuracy. Our theoretical analysis
        provides convergence guarantees under standard assumptions.
        """
    }

    print("Testing review generation...")
    results = generate_all_reviews(test_paper, models=["gpt4"], delay_between_calls=0)

    for model_id, result in results.items():
        print(f"\n--- {result.model_name} ---")
        if result.success:
            review = result.parsed_review
            print(f"Rating: {review.get('rating')}")
            print(f"Recommendation: {review.get('recommendation')}")
            print(f"Summary: {review.get('summary')}")
        else:
            print(f"Error: {result.error_message}")
