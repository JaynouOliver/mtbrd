"""
Visual Similarity Search - Find materials that look similar.
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

if not SUPABASE_URL or not API_KEY:
    st.error("Missing environment variables: SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env file")
    st.stop()

# Sample MATERIAL products
SAMPLE_PRODUCTS = [
    {"id": "6a9f9346-5e0c-4011-ba52-d3d95975ad05", "name": "White Carrara Marble", "supplier": "MSI Surfaces", "image": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-6a9f9346-5e0c-4011-ba52-d3d95975ad05_color_original_e1569a96-7b9b-4892-965e-6302c74d556c.jpg"},
    {"id": "b474d07068cd48728308bcf0d3fbe386", "name": "Arabescato Marble", "supplier": "Cosentino", "image": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-b474d07068cd48728308bcf0d3fbe386_color_original_268764fa-6344-4b6d-91d0-eaff3accd3c2.jpg"},
    {"id": "fdb1c6f2-bb8a-4ed1-a99c-bfb175299958", "name": "Wallowa Pine Wood", "supplier": "Arauco", "image": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-fdb1c6f2-bb8a-4ed1-a99c-bfb175299958_color_original_628a0c7c-9356-4246-8fda-e7f00a90f6a4.jpg"},
    {"id": "7ca456d0-7f8d-49fa-88f0-307bb82a4858", "name": "Step by Step Carpet", "supplier": "J+J Flooring", "image": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-7ca456d0-7f8d-49fa-88f0-307bb82a4858_color_original_ad29b924-de0b-4e5d-8df0-e9621360f08c.jpg"},
    {"id": "Material-3ipz40jcjp", "name": "Waterfall Succulent", "supplier": "Drop It Modern", "image": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-3ipz40jcjp/color.jpg"},
    {"id": "Material-04389", "name": "Hallingdal Fabric", "supplier": "Kvadrat", "image": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-r93axeycwf/Material-r93axeycwf_color.jpg"},
    {"id": "03037d26-6abe-4798-9f60-e040a4db4a61", "name": "Levine Rug", "supplier": "Stark", "image": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-03037d26-6abe-4798-9f60-e040a4db4a61_color_original_d0c68cf6-5a15-49d6-8aa5-cfabc30180a0.jpg"},
    {"id": "ee6c7142-c12f-4bca-9a24-1ae7281b42af", "name": "Luxurious Wallcovering", "supplier": "Koroseal", "image": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-ee6c7142-c12f-4bca-9a24-1ae7281b42af_color_original_4020ee98-e3e7-4e81-af39-be374c7d2812.jpg"},
    {"id": "ef806014-1132-47fc-a511-bf5116e17275", "name": "Rosegold Glyph Tile", "supplier": "Florim", "image": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-ef806014-1132-47fc-a511-bf5116e17275_color_original_28b6e7ca-3023-4f2e-ad16-a66c2536e640.jpg"},
    {"id": "Material-00200", "name": "Schwarzwald Verdure", "supplier": "Dedar", "image": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-00200/0faognlurj_color.jpg"},
    {"id": "15766709-9d0f-4796-88bd-81ed1fe9d43a", "name": "House of Tweed", "supplier": "Koroseal", "image": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-15766709-9d0f-4796-88bd-81ed1fe9d43a_color_original_39c020c0-6fae-4658-be30-545a84582369.jpg"},
    {"id": "Material-00303", "name": "Terracotta Tile", "supplier": "Tile", "image": "https://storage.googleapis.com/mattoboard-b8284.appspot.com/gltf-materials/Material-00303/n5ncrcyv1f_color.jpg"},
]

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
    .supplier-text { color: #94a3b8; font-size: 12px; }
    .header-gradient { background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .stButton > button { background: linear-gradient(135deg, #3b82f6, #2563eb) !important; color: white !important; border: none !important; font-weight: 600 !important; border-radius: 8px !important; }
    hr { border-color: #334155; }
</style>
""", unsafe_allow_html=True)


# ============= OPTIMIZATION 1: Session Pooling =============
@st.cache_resource
def get_session():
    """Reusable session with connection pooling."""
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {API_KEY}",
        "apikey": API_KEY,
        "Content-Type": "application/json"
    })
    return session


# ============= OPTIMIZATION 2: Client-Side Deduplication =============
def dedupe_by_product_group(results: list, limit: int) -> list:
    """
    Fast client-side deduplication by product_group_id.
    ~1-2ms vs 100-150ms in SQL.
    """
    seen_groups = set()
    unique_results = []
    
    for item in results:
        # Use product_group_id if available, otherwise use id
        group_key = item.get('product_group_id') or item['id']
        
        if group_key not in seen_groups:
            seen_groups.add(group_key)
            unique_results.append(item)
            
            if len(unique_results) >= limit:
                break
    
    return unique_results


# ============= OPTIMIZATION 3: Response Caching =============
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


@st.cache_data(ttl=300)
def fetch_similar_materials_cached(product_id: str, limit: int = 12) -> tuple:
    """
    Fetch similar materials with:
    - Ultra-simple HNSW query (v4) - ~5ms DB time
    - Client-side deduplication - ~1ms
    - Response caching - instant for repeated queries
    
    Returns (data, latency_ms)
    """
    session = get_session()
    start = time.time()
    
    try:
        # Over-fetch for client-side deduplication (2x limit)
        over_fetch_count = limit * 2
        
        response = session.post(
            f"{SUPABASE_URL}/rest/v1/rpc/search_similar_v4",
            json={"query_id": product_id, "match_cnt": over_fetch_count},
            timeout=12
        )
        
        api_latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            results = response.json()
            
            # Client-side deduplication (~1-2ms)
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
                        "thumbnail_url": r["image_url"],
                        "similarity_score": r["similarity"]
                    }
                    for r in unique_results if r.get("image_url")
                ],
                "count": len(unique_results),
                "api_latency": api_latency,
                "dedup_time": dedup_time
            }
            return data, total_latency
        return None, api_latency
    except Exception as e:
        latency = (time.time() - start) * 1000
        return None, latency


# ============= UI =============
st.markdown("<h1 class='header-gradient'>Material Similarity Search</h1>", unsafe_allow_html=True)
st.markdown("Select a material or enter a product ID to find visually similar options.")

st.markdown("---")

# Custom Product ID Input
st.subheader("Search by Product ID")
col_input, col_btn = st.columns([3, 1])
with col_input:
    custom_id = st.text_input("Enter Product ID:", placeholder="e.g. 6a9f9346-5e0c-4011-ba52-d3d95975ad05", label_visibility="collapsed")
with col_btn:
    if st.button("Search", type="primary", use_container_width=True):
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
                    "supplier": "ID not found",
                    "image": None
                }

st.markdown("---")

# Sample Materials Grid
st.subheader("Or Select a Sample Material")
cols = st.columns(4)
for i, product in enumerate(SAMPLE_PRODUCTS):
    with cols[i % 4]:
        st.image(product["image"], use_container_width=True)
        st.markdown(f"**{product['name'][:22]}**")
        st.markdown(f"<span class='supplier-text'>{product['supplier']}</span>", unsafe_allow_html=True)
        if st.button("Select", key=f"btn_{product['id']}", use_container_width=True):
            st.session_state.selected_product = product

# Results Section
if "selected_product" in st.session_state:
    selected_product = st.session_state.selected_product
    
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("<span class='query-badge'>Selected Material</span>", unsafe_allow_html=True)
        if selected_product.get("image"):
            st.image(selected_product["image"], use_container_width=True)
        else:
            st.info(f"ID: {selected_product['id']}")
        st.markdown(f"**{selected_product['name']}**")
        st.caption(f"Supplier: {selected_product['supplier']}")
    
    with col2:
        st.subheader("Visually Similar Materials")
        
        with st.spinner("Finding similar materials..."):
            results, latency = fetch_similar_materials_cached(selected_product["id"], limit=12)
        
        # Show latency breakdown
        if latency < 300:
            latency_color = "#10b981"  # Green
            status = "✅"
        elif latency < 500:
            latency_color = "#f59e0b"  # Yellow
            status = "⚠️"
        else:
            latency_color = "#ef4444"  # Red
            status = "❌"
        
        latency_info = f"{status} Total: <span style='color:{latency_color}'>{latency:.0f}ms</span>"
        if results and "api_latency" in results:
            latency_info += f" (API: {results['api_latency']:.0f}ms, Dedup: {results['dedup_time']:.1f}ms)"
        
        st.markdown(f"<span class='latency-badge'>⚡ {latency_info}</span>", unsafe_allow_html=True)
        
        if results and results.get("data"):
            data = results["data"]
            st.success(f"Found {len(data)} unique similar materials (deduplicated)")
            
            result_cols = st.columns(4)
            for j, item in enumerate(data):
                with result_cols[j % 4]:
                    img_url = item.get("thumbnail_url")
                    if img_url:
                        st.image(img_url, use_container_width=True)
                    else:
                        st.info("No image")
                    
                    score = item.get("similarity_score", 0)
                    st.markdown(f"<span class='score-badge'>{score * 100:.1f}% match</span>", unsafe_allow_html=True)
                    st.markdown(f"**{item.get('name', 'Unknown')[:28]}**")
                    st.markdown("---")
        else:
            st.warning("No similar materials found.")
else:
    st.info("Click 'Select' on any material above to find similar options.")

st.markdown("---")
st.caption("Powered by Voyage AI multimodal embeddings | Optimized: Session pooling + Client-side dedup + Caching")
