"""
Visual Similarity Search - Compare Voyage vs DINOv2 embeddings.
Optimized for <300ms response times with client-side deduplication.

Run: streamlit run app.py
"""

import streamlit as st
import requests
from typing import Optional
import time
import os
from dotenv import load_dotenv
load_dotenv()

# API Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
API_KEY = os.getenv("SUPABASE_ANON_KEY")

# Optional: use local backend for low-latency tests (e.g., http://localhost:8010)
LOCAL_API_URL = os.getenv("LOCAL_API_URL")  # if set, use this instead of Supabase REST

if not LOCAL_API_URL and (not SUPABASE_URL or not API_KEY):
    st.error("Missing environment variables: set LOCAL_API_URL for local backend, or SUPABASE_URL and SUPABASE_ANON_KEY for Supabase")
    st.stop()

# CSV-driven product pool (load from CSV, no images needed for selection)
CSV_PATH = os.getenv("PRODUCT_CSV_PATH", "Supabase Snippet Product Image Metadata Extract.csv")

def load_all_products_from_csv(csv_path: str) -> list:
    """Load all product ids/names from CSV."""
    import csv
    products = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("id")
                name = row.get("name") or pid
                if pid:
                    products.append({"id": pid, "name": name})
    except Exception as e:
        st.warning(f"Could not load CSV: {e}")
        products = []
    return products

# Load all products once at startup
ALL_PRODUCTS = load_all_products_from_csv(CSV_PATH)

def get_random_sample(products: list, n: int, seed: float) -> list:
    """Get n random unique products using seed for reproducibility."""
    import random
    rng = random.Random(seed)
    if len(products) <= n:
        items = products[:]
        rng.shuffle(items)
        return items
    return rng.sample(products, n)

# Model configurations
MODELS = {
    "voyage": {
        "name": "Voyage Multimodal-3.5",
        "rpc": "search_similar_v4",
        "description": "Multimodal (image+text) embeddings, good for semantic understanding",
        "dimensions": 1024,
        "color": "#3b82f6"
    },
    "dinov2": {
        "name": "DINOv2 (Meta)",
        "rpc": "search_similar_dinov2",
        "description": "Self-supervised vision model, excels at texture/surface details",
        "dimensions": 384,
        "color": "#8b5cf6"
    }
}

# Page config
st.set_page_config(page_title="Material Similarity Search", page_icon="🎨", layout="wide")

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    * { font-family: 'DM Sans', sans-serif; }
    .main { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
    h1, h2, h3 { color: #f8fafc !important; }
    .score-badge { background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: 600; }
    .query-badge { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; padding: 6px 14px; border-radius: 20px; font-size: 14px; font-weight: 600; }
    .latency-badge { background: #1e293b; color: #94a3b8; padding: 4px 10px; border-radius: 8px; font-size: 12px; }
    .model-badge-voyage { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .model-badge-dinov2 { background: linear-gradient(135deg, #8b5cf6, #7c3aed); color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .supplier-text { color: #94a3b8; font-size: 12px; }
    .header-gradient { background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .stButton > button { background: linear-gradient(135deg, #3b82f6, #2563eb) !important; color: white !important; border: none !important; font-weight: 600 !important; border-radius: 8px !important; }
    hr { border-color: #334155; }
    .comparison-header { background: linear-gradient(135deg, #1e293b, #334155); padding: 12px 16px; border-radius: 12px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)


# ============= Session Pooling =============
@st.cache_resource
def get_session():
    """Reusable session with connection pooling."""
    session = requests.Session()
    if not LOCAL_API_URL:
        session.headers.update({
            "Authorization": f"Bearer {API_KEY}",
            "apikey": API_KEY,
            "Content-Type": "application/json"
        })
    return session


# ============= Client-Side Deduplication =============
def dedupe_by_product_group(results: list, limit: int) -> list:
    """Fast client-side deduplication by product_group_id. ~1-2ms"""
    seen_groups = set()
    unique_results = []
    
    for item in results:
        group_key = item.get('product_group_id') or item['id']
        if group_key not in seen_groups:
            seen_groups.add(group_key)
            unique_results.append(item)
            if len(unique_results) >= limit:
                break
    
    return unique_results


# ============= API Functions =============
@st.cache_data(ttl=3600)
def fetch_product_details_cached(product_id: str) -> Optional[dict]:
    """Fetch product details with caching."""
    session = get_session()
    try:
        response = session.post(
            f"{SUPABASE_URL}/rest/v1/rpc/get_material_details",
            json={"product_id": product_id},
            timeout=10
        )
        if response.status_code == 200:
            results = response.json()
            if results and len(results) > 0:
                return results[0]
        return None
    except:
        return None


def fetch_similar_materials(product_id: str, model: str, limit: int = 10) -> tuple:
    """
    Fetch similar materials using specified model.
    Returns (data, latency_ms, error_msg)
    """
    session = get_session()
    start = time.time()
    
    model_config = MODELS.get(model, MODELS["voyage"])
    rpc_name = model_config["rpc"]
    
    try:
        # Over-fetch 100 neighbors per CTO spec to handle variant problem
        # After dedup by product_group_id, we return `limit` unique results
        over_fetch_count = max(100, limit * 10)

        if LOCAL_API_URL:
            # Local backend (GET with query params)
            endpoint = "voyage" if model == "voyage" else "dinov2"
            resp = session.get(
                f"{LOCAL_API_URL}/{endpoint}",
                params={"query_id": product_id, "limit": over_fetch_count},
                timeout=8,
            )
            api_latency = (time.time() - start) * 1000
            if resp.status_code != 200:
                return None, api_latency, f"Local API Error {resp.status_code}: {resp.text[:200]}"
            payload = resp.json()
            results = payload.get("results", [])
            api_latency = payload.get("benchmark_ms", api_latency)
        else:
            # Supabase REST RPC
            resp = session.post(
                f"{SUPABASE_URL}/rest/v1/rpc/{rpc_name}",
                json={"query_id": product_id, "match_cnt": over_fetch_count},
                timeout=12
            )
            api_latency = (time.time() - start) * 1000
            if resp.status_code != 200:
                return None, api_latency, f"API Error {resp.status_code}: {resp.text[:200]}"
            results = resp.json()

        # Client-side deduplication
        dedup_start = time.time()
        unique_results = dedupe_by_product_group(results, limit)
        dedup_time = (time.time() - dedup_start) * 1000

        total_latency = api_latency + dedup_time

        data = {
            "data": [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "product_type": r["product_type"],
                    "product_group_id": r.get("product_group_id"),
                    "thumbnail_url": r.get("image_url"),
                    "similarity_score": r.get("similarity")
                }
                for r in unique_results if r.get("image_url")
            ],
            "count": len(unique_results),
            "api_latency": api_latency,
            "dedup_time": dedup_time,
            "model": model
        }
        return data, total_latency, None
    except Exception as e:
        latency = (time.time() - start) * 1000
        return None, latency, str(e)


# ============= UI =============
st.markdown("<h1 class='header-gradient'>Material Similarity Search</h1>", unsafe_allow_html=True)
st.markdown("Compare embedding models: **Voyage Multimodal-3.5** vs **DINOv2 (Meta)**")

st.markdown("---")

# Model Selection
col_model, col_mode = st.columns([2, 2])
with col_model:
    model_choice = st.radio(
        "Embedding Model:",
        options=["voyage", "dinov2", "compare"],
        format_func=lambda x: {
            "voyage": "Voyage Multimodal-3.5 (1024d)",
            "dinov2": "DINOv2 by Meta (384d)",
            "compare": "Compare Both Side-by-Side"
        }[x],
        horizontal=True
    )

with col_mode:
    if model_choice == "voyage":
        st.info("Voyage: Multimodal embeddings, good for semantic/style understanding")
    elif model_choice == "dinov2":
        st.info("DINOv2: Self-supervised vision, excels at texture/surface details")
    else:
        st.info("Side-by-side comparison of both models")

st.markdown("---")

# Custom Product ID Input
st.subheader("Search by Product ID")
col_input, col_btn = st.columns([3, 1])
with col_input:
    custom_id = st.text_input("Enter Product ID:", placeholder="e.g. 6a9f9346-5e0c-4011-ba52-d3d95975ad05", label_visibility="collapsed")
with col_btn:
    if st.button("Search", type="primary", width="stretch"):
        if custom_id.strip():
            details = fetch_product_details_cached(custom_id.strip())
            if details:
                st.session_state.selected_product = {
                    "id": details["id"],
                    "name": details.get("name") or f"Product {custom_id.strip()[:8]}...",
                    "supplier": details.get("supplier") or "Unknown",
                    "image": details.get("image_url")
                }
            else:
                st.session_state.selected_product = {
                    "id": custom_id.strip(),
                    "name": f"Product {custom_id.strip()[:8]}...",
                    "supplier": "ID lookup",
                    "image": None
                }

st.markdown("---")

# Sample Materials List (no images, random 20 from CSV)
st.subheader("Or Select a Sample Material")

# Initialize shuffle seed if not present
if "sample_seed" not in st.session_state:
    import time as _time
    st.session_state.sample_seed = _time.time()

# Reshuffle button
if st.button("Reshuffle Product IDs"):
    import time as _time
    st.session_state.sample_seed = _time.time()
    st.rerun()

# Get 20 random products based on current seed
sample_products = get_random_sample(ALL_PRODUCTS, 20, st.session_state.sample_seed)

if not sample_products:
    st.warning("No products found in CSV. Check the file path.")
else:
    # Display as a simple list (no images)
    for product in sample_products:
        cols_row = st.columns([4, 1])
        with cols_row[0]:
            st.markdown(f"**{product['name'][:50]}**")
            st.caption(f"ID: `{product['id']}`")
        with cols_row[1]:
            if st.button("Select", key=f"btn_{product['id']}"):
                st.session_state.selected_product = {
                    "id": product["id"],
                    "name": product["name"],
                    "supplier": "CSV",
                    "image": None  # No image in selection, will show in results
                }
                st.rerun()

# Results Section
if "selected_product" in st.session_state:
    selected_product = st.session_state.selected_product
    
    st.markdown("---")
    
    # Query product info
    col_query, col_results = st.columns([1, 4])
    
    with col_query:
        st.markdown("<span class='query-badge'>Query Material</span>", unsafe_allow_html=True)
        if selected_product.get("image"):
            st.image(selected_product["image"], width="stretch")
        else:
            st.info(f"ID: {selected_product['id'][:20]}...")
        st.markdown(f"**{selected_product['name']}**")
        st.caption(f"Supplier: {selected_product['supplier']}")
    
    with col_results:
        if model_choice == "compare":
            # Side-by-side comparison
            st.subheader("Side-by-Side Comparison")
            
            col_v, col_d = st.columns(2)
            
            with col_v:
                st.markdown("<div class='comparison-header'><span class='model-badge-voyage'>Voyage Multimodal-3.5</span></div>", unsafe_allow_html=True)
                with st.spinner("Searching with Voyage..."):
                    results_v, latency_v, error_v = fetch_similar_materials(selected_product["id"], "voyage", 8)
                
                if error_v:
                    st.error(f"Voyage error: {error_v}")
                elif results_v and results_v.get("data"):
                    latency_color = "#10b981" if latency_v < 300 else "#f59e0b" if latency_v < 500 else "#ef4444"
                    st.markdown(f"<span class='latency-badge'>⚡ <span style='color:{latency_color}'>{latency_v:.0f}ms</span></span>", unsafe_allow_html=True)
                    
                    for item in results_v["data"]:
                        cols_item = st.columns([1, 2])
                        with cols_item[0]:
                            if item.get("thumbnail_url"):
                                st.image(item["thumbnail_url"], width=80)
                        with cols_item[1]:
                            score = item.get("similarity_score", 0)
                            st.markdown(f"<span class='score-badge'>{score * 100:.1f}%</span>", unsafe_allow_html=True)
                            st.markdown(f"**{item.get('name', 'Unknown')[:25]}**")
                else:
                    st.warning("No results from Voyage")
            
            with col_d:
                st.markdown("<div class='comparison-header'><span class='model-badge-dinov2'>DINOv2 (Meta)</span></div>", unsafe_allow_html=True)
                with st.spinner("Searching with DINOv2..."):
                    results_d, latency_d, error_d = fetch_similar_materials(selected_product["id"], "dinov2", 8)
                
                if error_d:
                    st.error(f"DINOv2 error: {error_d}")
                elif results_d and results_d.get("data"):
                    latency_color = "#10b981" if latency_d < 300 else "#f59e0b" if latency_d < 500 else "#ef4444"
                    st.markdown(f"<span class='latency-badge'>⚡ <span style='color:{latency_color}'>{latency_d:.0f}ms</span></span>", unsafe_allow_html=True)
                    
                    for item in results_d["data"]:
                        cols_item = st.columns([1, 2])
                        with cols_item[0]:
                            if item.get("thumbnail_url"):
                                st.image(item["thumbnail_url"], width=80)
                        with cols_item[1]:
                            score = item.get("similarity_score", 0)
                            st.markdown(f"<span class='score-badge'>{score * 100:.1f}%</span>", unsafe_allow_html=True)
                            st.markdown(f"**{item.get('name', 'Unknown')[:25]}**")
                else:
                    st.warning("No results from DINOv2 (embedding may not exist for this product)")
        
        else:
            # Single model view
            model_config = MODELS[model_choice]
            st.subheader(f"Similar Materials ({model_config['name']})")
            
            badge_class = f"model-badge-{model_choice}"
            st.markdown(f"<span class='{badge_class}'>{model_config['name']} ({model_config['dimensions']}d)</span>", unsafe_allow_html=True)
            
            with st.spinner(f"Finding similar materials with {model_config['name']}..."):
                results, latency, error = fetch_similar_materials(selected_product["id"], model_choice, 12)
            
            if error:
                st.error(f"Error: {error}")
            elif results and results.get("data"):
                # Latency display
                latency_color = "#10b981" if latency < 300 else "#f59e0b" if latency < 500 else "#ef4444"
                status = "✅" if latency < 300 else "⚠️" if latency < 500 else "❌"
                
                latency_info = f"{status} Total: <span style='color:{latency_color}'>{latency:.0f}ms</span>"
                latency_info += f" (API: {results['api_latency']:.0f}ms, Dedup: {results['dedup_time']:.1f}ms)"
                
                st.markdown(f"<span class='latency-badge'>⚡ {latency_info}</span>", unsafe_allow_html=True)
                st.success(f"Found {len(results['data'])} unique similar materials")
                
                result_cols = st.columns(4)
                for j, item in enumerate(results["data"]):
                    with result_cols[j % 4]:
                        img_url = item.get("thumbnail_url")
                        if img_url:
                            st.image(img_url, width="stretch")
                        else:
                            st.info("No image")
                        
                        score = item.get("similarity_score", 0)
                        st.markdown(f"<span class='score-badge'>{score * 100:.1f}% match</span>", unsafe_allow_html=True)
                        st.markdown(f"**{item.get('name', 'Unknown')[:28]}**")
                        st.markdown("---")
            else:
                if model_choice == "dinov2":
                    st.warning("No DINOv2 embedding found for this product. Try another material or use Voyage.")
                else:
                    st.warning("No similar materials found.")

else:
    st.info("Click 'Select' on any material above to find similar options.")

st.markdown("---")
st.caption("Embedding Models: Voyage Multimodal-3.5 (1024d) | DINOv2-ViT-S/14 (384d) | Optimized: Session pooling + Client-side dedup")
