import os

import voyageai
import PIL
import requests
from dotenv import load_dotenv

load_dotenv()
vo = voyageai.Client()
# This will automatically use the environment variable VOYAGE_API_KEY.

def embed_image_path(image_path: str, context_text: str = None) -> list:
    """
    Given an image file path, returns its Voyage multimodal embedding (1024-D vector).
    
    Uses input_type="document" to match database embeddings format.
    Text context is optional - if None, embeds only the image to match DB embeddings without text.
    
    Args:
        image_path: Path to image file
        context_text: Optional text context (None = image only, matches DB format)
    
    Returns:
        1024-dimensional embedding vector
    """
    img = PIL.Image.open(image_path)
    
    # Embed image only (no text) to match database format
    # Note: input_type is optional. Database uses "document", but embeddings are compatible.
    # You can test with/without input_type to see which gives better similarity scores.
    if context_text:
        inputs = [[context_text, img]]
    else:
        inputs = [[img]]
    
    # Using input_type="document" to match database embeddings format
    # Remove input_type parameter if you want to test without it
    result = vo.multimodal_embed(inputs, model="voyage-multimodal-3.5")
    return result.embeddings[0]


def embed_image_url(image_url: str, context_text: str = None) -> list:
    """
    Given an image URL, returns its Voyage multimodal embedding (1024-D vector).
    
    This mirrors `embed_image_path` but passes the remote image URL directly to
    Voyage's multimodal embeddings endpoint using an `image_url` content block.
    
    Args:
        image_url: Publicly accessible URL to the image.
        context_text: Optional text context (None = image only).
    
    Returns:
        1024-dimensional embedding vector (list of floats).
    """
    # Build multimodal content: optional text + image URL
    content = []
    if context_text:
        content.append({"type": "text", "text": context_text})
    content.append({"type": "image_url", "image_url": image_url})

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY environment variable is not set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": [{"content": content}],
        "model": "voyage-multimodal-3.5",
        "input_type": "document",
        "truncation": True,
    }

    response = requests.post(
        "https://api.voyageai.com/v1/multimodalembeddings",
        headers=headers,
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Voyage API error {response.status_code}: {response.text[:300]}"
        )

    data = response.json()
    if not data.get("data"):
        raise RuntimeError("Voyage API response does not contain 'data' field.")

    embedding = data["data"][0].get("embedding")
    if embedding is None:
        raise RuntimeError("Voyage API response does not contain 'embedding'.")

    return embedding



if __name__ == "__main__":
    # Simple manual test using a direct image URL
    test_url = (
        "https://img.freepik.com/free-photo/"
        "beautiful-view-sunset-sea_23-2148019892.jpg?size=626&ext=jpg"
    )
    emb = embed_image_url(test_url)
    print("Embedding length:", len(emb))
    print("Embedding values:", emb)