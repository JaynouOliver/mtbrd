#!/usr/bin/env python3
"""
Test similarity search latency on Mumbai read replica.
Uses direct PostgreSQL connection via session pooler.
"""

import psycopg2
import time
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Connection config
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port", "5432")
DBNAME = os.getenv("dbname")

# Test product IDs (with both embeddings)
TEST_PRODUCTS = [
    ("3ddac862-5cab-408e-8f48-798133bbc579", "Adjust"),
    ("3dd6e3af-fc78-4dd3-a2a1-ec8f5d9a6c8e", "Mystone Ceppo di Gré"),
    ("3ddc4e57-fd96-4888-a736-17112b2bffd3", "Joy Squared 12X48"),
]


def search_similar_voyage(cursor, product_id: str, match_cnt: int = 24):
    """Execute Voyage similarity search via RPC function."""
    cursor.execute(
        "SELECT * FROM search_similar_v4(%s, %s)",
        (product_id, match_cnt)
    )
    return cursor.fetchall()


def search_similar_dinov2(cursor, product_id: str, match_cnt: int = 24):
    """Execute DINOv2 similarity search via RPC function."""
    cursor.execute(
        "SELECT * FROM search_similar_dinov2(%s, %s)",
        (product_id, match_cnt)
    )
    return cursor.fetchall()


def test_latency():
    """Test similarity search latency."""
    print("=" * 60)
    print("MUMBAI READ REPLICA LATENCY TEST")
    print("=" * 60)
    print(f"\nConnecting to: {HOST}:{PORT}")
    
    try:
        # Connect
        start = time.time()
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        connect_time = (time.time() - start) * 1000
        print(f"✓ Connected in {connect_time:.1f}ms")
        
        cursor = connection.cursor()
        
        # Warmup query
        cursor.execute("SELECT 1")
        cursor.fetchone()
        
        print("\n" + "-" * 60)
        print("VOYAGE (search_similar_v4) - 1024d embeddings")
        print("-" * 60)
        
        voyage_times = []
        for product_id, name in TEST_PRODUCTS:
            times = []
            for run in range(5):
                start = time.time()
                results = search_similar_voyage(cursor, product_id)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
            
            avg = sum(times) / len(times)
            min_t = min(times)
            max_t = max(times)
            voyage_times.extend(times)
            print(f"  {name[:25]:<25} | Avg: {avg:>6.1f}ms | Min: {min_t:>6.1f}ms | Max: {max_t:>6.1f}ms | Results: {len(results)}")
        
        print(f"\n  VOYAGE OVERALL: Avg={sum(voyage_times)/len(voyage_times):.1f}ms, Min={min(voyage_times):.1f}ms, Max={max(voyage_times):.1f}ms")
        
        print("\n" + "-" * 60)
        print("DINOV2 (search_similar_dinov2) - 384d embeddings")
        print("-" * 60)
        
        dinov2_times = []
        for product_id, name in TEST_PRODUCTS:
            times = []
            for run in range(5):
                start = time.time()
                results = search_similar_dinov2(cursor, product_id)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
            
            avg = sum(times) / len(times)
            min_t = min(times)
            max_t = max(times)
            dinov2_times.extend(times)
            print(f"  {name[:25]:<25} | Avg: {avg:>6.1f}ms | Min: {min_t:>6.1f}ms | Max: {max_t:>6.1f}ms | Results: {len(results)}")
        
        print(f"\n  DINOV2 OVERALL: Avg={sum(dinov2_times)/len(dinov2_times):.1f}ms, Min={min(dinov2_times):.1f}ms, Max={max(dinov2_times):.1f}ms")
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        voyage_avg = sum(voyage_times) / len(voyage_times)
        dinov2_avg = sum(dinov2_times) / len(dinov2_times)
        print(f"  Voyage (1024d):  {voyage_avg:.1f}ms avg")
        print(f"  DINOv2 (384d):   {dinov2_avg:.1f}ms avg")
        print(f"  Improvement:     {((voyage_avg - dinov2_avg) / voyage_avg * 100):.1f}% faster with DINOv2")
        
        target = 300
        voyage_status = "✓ PASS" if voyage_avg < target else "✗ FAIL"
        dinov2_status = "✓ PASS" if dinov2_avg < target else "✗ FAIL"
        print(f"\n  Target: <{target}ms")
        print(f"  Voyage: {voyage_status} ({voyage_avg:.1f}ms)")
        print(f"  DINOv2: {dinov2_status} ({dinov2_avg:.1f}ms)")
        
        cursor.close()
        connection.close()
        print("\n✓ Connection closed.")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    test_latency()



