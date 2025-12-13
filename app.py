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


def fetch_similar_materials(product_id: str, model: str, limit: int = 12) -> tuple:
    """
    Fetch similar materials using specified model.
    Returns (data, latency_ms, error_msg)
    """
    session = get_session()
    start = time.time()
    
    model_config = MODELS.get(model, MODELS["voyage"])
    rpc_name = model_config["rpc"]
    
    try:
        over_fetch_count = limit * 2
        
        response = session.post(
            f"{SUPABASE_URL}/rest/v1/rpc/{rpc_name}",
            json={"query_id": product_id, "match_cnt": over_fetch_count},
            timeout=12
        )
        
        api_latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            results = response.json()
            
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
                        "thumbnail_url": r["image_url"],
                        "similarity_score": r["similarity"]
                    }
                    for r in unique_results if r.get("image_url")
                ],
                "count": len(unique_results),
                "api_latency": api_latency,
                "dedup_time": dedup_time,
                "model": model
            }
            return data, total_latency, None
        else:
            error_text = response.text[:200]
            return None, api_latency, f"API Error {response.status_code}: {error_text}"
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

# Sample Materials Grid
st.subheader("Or Select a Sample Material")
cols = st.columns(4)
for i, product in enumerate(SAMPLE_PRODUCTS):
    with cols[i % 4]:
        st.image(product["image"], width="stretch")
        st.markdown(f"**{product['name'][:22]}**")
        st.markdown(f"<span class='supplier-text'>{product['supplier']}</span>", unsafe_allow_html=True)
        if st.button("Select", key=f"btn_{product['id']}", width="stretch"):
            st.session_state.selected_product = product

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
