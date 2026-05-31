import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.logger import get_logger

logger = get_logger(__name__)


class CLIPEmbeddingGenerator:
    def __init__(self):
        logger.info("Initializing CLIPEmbeddingGenerator")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            if self.device == "cuda":
                cuda_device_count = torch.cuda.device_count()
                cuda_device_name = torch.cuda.get_device_name(0)
                logger.info(
                    f"CUDA available - Device count: {cuda_device_count}, Device: {cuda_device_name}"
                )

            logger.info("Loading CLIP model: openai/clip-vit-base-patch32")
            # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            #     self.device
            # )

            self.model = (
                CLIPModel
                .from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir="/tmp/hf_cache"
                ).to(self.device)
)
            
            logger.info(f"Model loaded successfully on {self.device}")

            logger.info("Loading CLIP processor")
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )

            self.processor = (
                CLIPProcessor
                .from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir="/tmp/hf_cache")
)
            logger.info("Processor loaded successfully")
            logger.info("CLIPEmbeddingGenerator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CLIPEmbeddingGenerator: {str(e)}")
            raise

    @torch.no_grad()
    def generate_embedding(self, image_path):
        try:
            logger.debug(f"Generating embedding for image: {image_path}")

            image = Image.open(image_path).convert("RGB")
            logger.debug(f"Image loaded successfully, size: {image.size}")

            inputs = self.processor(images=image, return_tensors="pt")
            logger.debug(f"Image preprocessed successfully")

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            embedding = self.model.get_image_features(**inputs)
            logger.debug(
                f"Model inference completed, embedding shape: {embedding.shape}"
            )

            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            logger.debug(f"Embedding normalized")

            return embedding.cpu().numpy()[0]
        except Image.UnidentifiedImageError as e:
            logger.error(f"Invalid image file {image_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate embedding for {image_path}: {str(e)}")
            raise
