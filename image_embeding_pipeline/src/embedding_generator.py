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

            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir="/tmp/hf_cache"
            ).to(self.device)

            logger.info(f"Model loaded successfully on {self.device}")

            logger.info("Loading CLIP processor")
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )

            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir="/tmp/hf_cache"
            )
            logger.info("Processor loaded successfully")
            logger.info("CLIPEmbeddingGenerator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CLIPEmbeddingGenerator: {str(e)}")
            raise

    @staticmethod
    def create_image_batches(image_paths, batch_size):
        """
        Create batches of image paths for GPU processing.

        Args:
            image_paths: List of image file paths
            batch_size: Size of each batch

        Returns:
            List of batches (each batch is a list of image paths)
        """
        batches = []
        for i in range(0, len(image_paths), batch_size):
            batches.append(image_paths[i : i + batch_size])
        return batches

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

    @torch.no_grad()
    def generate_batch_embeddings(self, image_paths):
        """
        Generate embeddings for multiple images in a single GPU batch.

        Args:
            image_paths: List of image file paths

        Returns:
            List of normalized embeddings (numpy arrays)
        """
        try:
            logger.debug(f"Generating batch embeddings for {len(image_paths)} images")

            images = []
            valid_paths = []

            # Load all images
            for image_path in image_paths:
                try:
                    image = Image.open(image_path).convert("RGB")
                    images.append(image)
                    valid_paths.append(image_path)
                except Image.UnidentifiedImageError as e:
                    logger.warning(f"Invalid image file {image_path}: {str(e)}")
                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {str(e)}")

            if len(images) == 0:
                logger.warning("No valid images in batch")
                return []

            logger.debug(f"Loaded {len(images)} valid images out of {len(image_paths)}")

            # Process all images together
            inputs = self.processor(images=images, return_tensors="pt")
            logger.debug(f"Batch preprocessed successfully")

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            embeddings = self.model.get_image_features(**inputs)
            logger.debug(
                f"Model inference completed, embeddings shape: {embeddings.shape}"
            )

            # Normalize embeddings
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            logger.debug(f"Embeddings normalized")

            # Convert to numpy and return as list
            embeddings_np = embeddings.cpu().numpy()
            logger.debug(
                f"Batch embeddings generated successfully: {embeddings_np.shape}"
            )

            return embeddings_np.tolist(), valid_paths
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {str(e)}")
            raise
