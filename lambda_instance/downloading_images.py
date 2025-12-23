import csv
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import psycopg2
import requests
from requests.adapters import HTTPAdapter, Retry

# Config
BATCH_SIZE = 1000
LIMIT = 30000
OUT_DIR = Path("downloads/materials")
ORIG_DIR = OUT_DIR / "original"
UPS_DIR = OUT_DIR / "upscaled"
MANIFEST = OUT_DIR / "manifest.csv"
TIMEOUT = 15  # seconds per request
WORKERS = 16  # parallel download workers

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# DB connection details (override with env vars on the instance if available)
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "t1PAdg7zueX6pcvb")
DB_HOST = os.getenv("DB_HOST", "db.glfevldtqujajsalahxd-rr-ap-south-1-mxnjq.supabase.co")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

QUERY = """
    SELECT id,
           "productType",
           metadata->>'materialImageUrl' AS original_image,
           "materialData"->'files'->>'color_original' AS upscaled_image
    FROM public."productsV2"
    WHERE "productType" IN ('fixed material', 'material')
      AND "objectStatus" IN ('APPROVED', 'APPROVED_PRO')
      AND to_timestamp("updatedAt" / 1000) >= '2025-02-01'
      AND metadata->>'materialImageUrl' IS NOT NULL
      AND "materialData"->'files'->>'color_original' IS NOT NULL
    ORDER BY "updatedAt" DESC
    LIMIT %s
"""

def make_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=32, pool_maxsize=32)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def ensure_dirs():
    ORIG_DIR.mkdir(parents=True, exist_ok=True)
    UPS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_db_connection():
    """
    Direct PostgreSQL connection for the instance (no external helper).
    Fill the placeholder defaults above or set env vars on the instance.
    """
    return psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
    )

def fetch_rows(limit: int) -> List[Tuple]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(QUERY, (limit,))
            return cur.fetchall()
    finally:
        conn.close()

def determine_resume_index() -> int:
    """
    Find the highest consecutive index where both files already exist.
    Returns the count of completed items (0 if none).
    """
    idx = 1
    while True:
        idx_str = f"{idx:05d}"
        orig_path = ORIG_DIR / f"{idx_str}_orig.jpg"
        ups_path = UPS_DIR / f"{idx_str}_ups.jpg"
        if orig_path.exists() and ups_path.exists():
            idx += 1
            continue
        return idx - 1

def download_file(session, url: str, dest: Path) -> bool:
    try:
        resp = session.get(url, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"[warn] failed {url} -> {dest}: {e}")
        return False

def main():
    ensure_dirs()
    rows = fetch_rows(LIMIT)
    logging.info("Fetched %s rows", len(rows))

    completed = determine_resume_index()
    if completed:
        logging.info("Resuming from index %s (found %s completed files)", completed + 1, completed)
    else:
        logging.info("No completed downloads detected; starting fresh")

    manifest_mode = "a" if completed > 0 and MANIFEST.exists() else "w"

    sessions = [make_session() for _ in range(WORKERS)]
    total = len(rows)

    with MANIFEST.open(manifest_mode, newline="") as mf:
        writer = csv.writer(mf)
        if manifest_mode == "w":
            writer.writerow(["idx", "id", "productType", "original_path", "upscaled_path",
                             "original_url", "upscaled_url"])

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = []
            for i, row in enumerate(rows, 1):
                if i <= completed:
                    continue  # skip already-downloaded index
                pid, ptype, orig_url, ups_url = row
                idx_str = f"{i:05d}"
                sess = sessions[i % WORKERS]
                futures.append(executor.submit(
                    _download_one,
                    sess,
                    idx_str,
                    pid,
                    ptype,
                    orig_url,
                    ups_url,
                ))

            done = completed
            for fut in as_completed(futures):
                idx_str, pid, ptype, orig_path, ups_path, orig_url, ups_url = fut.result()
                writer.writerow([
                    idx_str, pid, ptype,
                    str(orig_path),
                    str(ups_path),
                    orig_url, ups_url
                ])
                done += 1
                if done % 500 == 0:
                    logging.info("Downloaded %s/%s...", done, total)
                    mf.flush()
                if done % 100 == 0:
                    logging.debug("Last completed idx: %s", idx_str)

    logging.info("Done. Manifest at %s", MANIFEST)

def _download_one(session, idx_str, pid, ptype, orig_url, ups_url):
    orig_path = ORIG_DIR / f"{idx_str}_orig.jpg"
    ups_path = UPS_DIR / f"{idx_str}_ups.jpg"
    ok_orig = download_file(session, orig_url, orig_path)
    ok_ups = download_file(session, ups_url, ups_path)
    return (
        idx_str,
        pid,
        ptype,
        orig_path if ok_orig else "",
        ups_path if ok_ups else "",
        orig_url,
        ups_url,
    )

if __name__ == "__main__":
    main()