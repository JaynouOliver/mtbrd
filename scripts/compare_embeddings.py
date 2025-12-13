#!/usr/bin/env python3
"""
Compare Voyage vs DINOv2 Embedding Quality
==========================================
Evaluates which embedding model produces better similarity results for materials.

Usage:
    python compare_embeddings.py [--sample-size 20]
"""

import os
import sys
import json
import argparse
import requests
from typing import Dict, List, Set, Tuple
from collections import defaultdict

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://glfevldtqujajsalahxd.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}


def fetch_sample_materials(limit: int = 20) -> List[Dict]:
    """Fetch sample materials that have both embeddings."""
    response = requests.get(
        f"{SUPABASE_URL}/rest/v1/products_search",
        params={
            "select": "id,name,supplier,materialData",
            "productType": "eq.material",
            "objectStatus": "in.(APPROVED,APPROVED_PRO)",
            "embedding_visual": "not.is.null",
            "embedding_dinov2": "not.is.null",
            "limit": limit
        },
        headers=HEADERS,
        timeout=30
    )
    return response.json() if response.status_code == 200 else []


def search_voyage(product_id: str, limit: int = 10) -> List[Dict]:
    """Search using Voyage embeddings (embedding_visual)."""
    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/search_similar_v4",
        json={"query_id": product_id, "match_cnt": limit},
        headers=HEADERS,
        timeout=30
    )
    return response.json() if response.status_code == 200 else []


def search_dinov2(product_id: str, limit: int = 10) -> List[Dict]:
    """Search using DINOv2 embeddings."""
    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/search_similar_dinov2",
        json={"query_id": product_id, "match_cnt": limit},
        headers=HEADERS,
        timeout=30
    )
    return response.json() if response.status_code == 200 else []


def compute_overlap(results1: List[Dict], results2: List[Dict]) -> Tuple[float, Set[str]]:
    """Compute Jaccard similarity between two result sets."""
    ids1 = {r["id"] for r in results1}
    ids2 = {r["id"] for r in results2}
    
    intersection = ids1 & ids2
    union = ids1 | ids2
    
    jaccard = len(intersection) / len(union) if union else 0
    return jaccard, intersection


def get_image_url(product: Dict) -> str:
    """Extract image URL from product."""
    if product.get("materialData"):
        files = product["materialData"].get("files", {})
        return files.get("color_original", "")
    return ""


def run_comparison(sample_size: int = 20, results_per_query: int = 10):
    """Run the comparison analysis."""
    print("=" * 70)
    print("VOYAGE vs DINOV2 EMBEDDING COMPARISON")
    print("=" * 70)
    
    # Fetch sample materials
    print(f"\nFetching {sample_size} sample materials with both embeddings...")
    samples = fetch_sample_materials(sample_size)
    
    if not samples:
        print("ERROR: No samples found with both embeddings. Run DINOv2 generation first.")
        return
    
    print(f"Found {len(samples)} samples\n")
    
    # Run comparisons
    overlaps = []
    voyage_only_counts = []
    dinov2_only_counts = []
    
    results_detail = []
    
    for i, sample in enumerate(samples, 1):
        product_id = sample["id"]
        product_name = sample.get("name", "Unknown")
        
        print(f"[{i}/{len(samples)}] {product_name[:50]}...")
        
        # Search with both models
        voyage_results = search_voyage(product_id, results_per_query)
        dinov2_results = search_dinov2(product_id, results_per_query)
        
        if not voyage_results or not dinov2_results:
            print(f"  Skipping - missing results")
            continue
        
        # Compute overlap
        jaccard, common = compute_overlap(voyage_results, dinov2_results)
        overlaps.append(jaccard)
        
        voyage_ids = {r["id"] for r in voyage_results}
        dinov2_ids = {r["id"] for r in dinov2_results}
        
        voyage_only = voyage_ids - dinov2_ids
        dinov2_only = dinov2_ids - voyage_ids
        
        voyage_only_counts.append(len(voyage_only))
        dinov2_only_counts.append(len(dinov2_only))
        
        print(f"  Overlap: {jaccard:.2%} | Voyage-only: {len(voyage_only)} | DINOv2-only: {len(dinov2_only)}")
        
        results_detail.append({
            "product_id": product_id,
            "product_name": product_name,
            "jaccard_overlap": jaccard,
            "common_count": len(common),
            "voyage_only_count": len(voyage_only),
            "dinov2_only_count": len(dinov2_only),
            "voyage_top3": [r["name"] for r in voyage_results[:3]],
            "dinov2_top3": [r["name"] for r in dinov2_results[:3]]
        })
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if overlaps:
        avg_overlap = sum(overlaps) / len(overlaps)
        avg_voyage_only = sum(voyage_only_counts) / len(voyage_only_counts)
        avg_dinov2_only = sum(dinov2_only_counts) / len(dinov2_only_counts)
        
        print(f"\nAverage Jaccard Overlap: {avg_overlap:.2%}")
        print(f"Average Voyage-only results: {avg_voyage_only:.1f}")
        print(f"Average DINOv2-only results: {avg_dinov2_only:.1f}")
        
        print("\n" + "-" * 70)
        print("INTERPRETATION:")
        print("-" * 70)
        
        if avg_overlap > 0.7:
            print("HIGH OVERLAP (>70%): Models find similar results.")
            print("Recommendation: Use either model, or the faster/cheaper one.")
        elif avg_overlap > 0.4:
            print("MODERATE OVERLAP (40-70%): Models capture different aspects.")
            print("Recommendation: Consider hybrid approach (weighted average).")
        else:
            print("LOW OVERLAP (<40%): Models are significantly different.")
            print("Recommendation: Test with users to determine which produces")
            print("more visually relevant results for your use case.")
        
        print("\n" + "-" * 70)
        print("TEXTURE-SPECIFIC INSIGHT:")
        print("-" * 70)
        print("DINOv2 typically excels at: Fine grain patterns, surface texture, material feel")
        print("Voyage typically excels at: Semantic similarity, style, context from metadata")
        
    # Save detailed results
    output = {
        "summary": {
            "sample_size": len(overlaps),
            "avg_jaccard_overlap": avg_overlap if overlaps else 0,
            "avg_voyage_only": avg_voyage_only if overlaps else 0,
            "avg_dinov2_only": avg_dinov2_only if overlaps else 0
        },
        "details": results_detail
    }
    
    output_file = "comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Voyage vs DINOv2 embeddings")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of samples to test")
    parser.add_argument("--results", type=int, default=10, help="Results per query")
    args = parser.parse_args()
    
    run_comparison(args.sample_size, args.results)



