import voyageai
import PIL
from dotenv import load_dotenv

load_dotenv()
vo = voyageai.Client()
# This will automatically use the environment variable VOYAGE_API_KEY.

def embed_image_path(image_path: str, context_text: str = "This is a room interior.") -> list:
    """
    Given an image file path, returns its Voyage multimodal embedding (1024-D vector).
    Optionally, a context_text can be supplied for multimodal queries.
    """
    img = PIL.Image.open(image_path)
    inputs = [[context_text, img]]
    result = vo.multimodal_embed(inputs, model="voyage-multimodal-3.5")
    return result.embeddings[0]

if __name__ == "__main__":
    test_image = "kitchen-calacatta-themis-01-v6.jpg"
    embedding = embed_image_path(test_image)
    print(embedding)