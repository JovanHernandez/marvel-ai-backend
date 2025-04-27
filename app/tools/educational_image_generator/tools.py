from datetime import datetime
import re
from typing import Dict, Optional
import uuid
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from app.services.logger import setup_logger
import os
import google.generativeai as genai
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel


# Import Google Cloud Storage libraries
try:
    from google.cloud import storage
    from google.oauth2 import service_account
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

logger = setup_logger(__name__)

class ImageGenerationOutput(BaseModel):
    image_url: str = Field(description="URL of the generated image")
    prompt_used: str = Field(description="The enhanced prompt used for generation")
    safety_status: str = Field(description="Status of safety checks")
    educational_context: dict = Field(description="Educational metadata about the generated image")

class ImageGenerator:
    def __init__(
            self,
            verbose: bool = False,
            project_id: Optional[str] = None,
            storage_bucket: Optional[str] = None,
    ):
        self.verbose = verbose
        self.project_id = project_id or os.getenv('PROJECT_ID')
        self.storage_bucket = storage_bucket or os.getenv('GCP_STORAGE_BUCKET')
        
        if not self.storage_bucket:
            raise ValueError("GCP_STORAGE_BUCKET environment variable is not set")
        
        # Configure Google Generative AI with API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
            
        genai.configure(api_key=api_key)
        
        # Initialize GCP Storage client
        try:
            self.storage_client = storage.Client()
            logger.info("Successfully initialized GCP Storage client")
        except Exception as e:
            logger.error(f"Failed to initialize GCP Storage client: {e}")
            raise
        
        # Initialize the Gemini model
        self.text_model = GoogleGenerativeAI(model="gemini-1.5-pro")
        
        # Read prompt template
        prompt_path = os.path.join(os.path.dirname(__file__), "prompt", "educational_image_prompt.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read().strip()

    def enhance_prompt(self, original_prompt: str, subject: str, grade_level: str) -> str:
        """Enhance the original prompt with educational context"""
        enhancement_prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["original_prompt", "subject", "grade_level"]
        )
        
        enhanced = self.text_model.invoke(
            enhancement_prompt.format(
                original_prompt=original_prompt,
                subject=subject,
                grade_level=grade_level
            )
        )
        return enhanced

    def check_safety(self, prompt: str) -> bool:
        """
        Validate prompt safety using the safety prompt template
        Returns:
            bool: True if safe, False if not safe
        """
        try:
            # Read safety prompt template
            safety_prompt_path = os.path.join(os.path.dirname(__file__), "prompt", "prompt_safety_prompt.txt")
            with open(safety_prompt_path, 'r', encoding='utf-8') as f:
                safety_template = f.read().strip()
            
            safety_prompt = PromptTemplate(
                template=safety_template,
                input_variables=["prompt"]
            )
            
            # Get model response - using text_model instead of model
            response = self.text_model.invoke(
                safety_prompt.format(
                    prompt=prompt
                )
            )
            
            # Clean and validate response
            response = response.strip().upper()
            if response not in ["SAFE", "NOT SAFE"]:
                logger.error(f"Invalid safety check response: {response}")
                raise ValueError(f"Invalid safety check response: {response}")
            
            if self.verbose:
                logger.info(f"Safety check result: {response}")
            
            return response == "SAFE"
        
        except Exception as e:
            logger.error(f"Error during safety check: {str(e)}")
            raise

    def generate_image(self, enhanced_prompt: str, lang: str = "en") -> str:
        """Generate image using Vertex AI's Imagen model and save to GCP bucket"""
        try:
            if self.verbose:
                logger.info(f"Generating image for prompt: {enhanced_prompt}")
            
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location="us-central1")
            image_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
            logger.info("Generating image using Vertex AI's imagen-3.0-generate-002")

            image_response = image_model.generate_images(
                prompt=enhanced_prompt,
                number_of_images=1,
                aspect_ratio="1:1",
                safety_filter_level="block_only_high",
                person_generation="allow_adult",
            )
            
            if not image_response:
                raise ValueError("No response received from the model")
            
            first_response = image_response[0]
            
            # Upload to GCP bucket
            gcp_url = self.upload_to_gcp_bucket(first_response._image_bytes, enhanced_prompt)
            if not gcp_url:
                raise ValueError("Failed to upload image to GCP bucket")
            
            logger.info(f"Successfully uploaded image to GCP bucket: {gcp_url}")
            return gcp_url
        
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise

    def upload_to_gcp_bucket(self, image_data: bytes, prompt: str) -> Optional[str]:
        """Upload an image to a GCP bucket and return the public URL."""
        if not GCP_AVAILABLE or not self.storage_client or not self.storage_bucket:
            if self.verbose:
                logger.info("GCP Storage not available or not configured, skipping upload")
            return None

        try:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            sanitized_prompt = re.sub(r'[^\w\s-]', '', prompt)[:30].strip().replace(' ', '_')
            filename = f"image_{timestamp}_{sanitized_prompt}_{unique_id}.png"

            # Upload image data to bucket
            bucket = self.storage_client.bucket(self.storage_bucket)
            blob = bucket.blob(f"generated_images/{filename}")
            blob.upload_from_string(image_data, content_type="image/png")

            public_url = blob.public_url
            if self.verbose:
                logger.info(f"Image uploaded to GCP bucket: {public_url}")

            return public_url

        except Exception as e:
            logger.error(f"Error uploading image to GCP bucket: {e}")
            return None
        
def generate_educational_image(
    prompt: str, 
    subject: str, 
    grade_level: str, 
    project_id: str,
    lang: str = "en", 
    verbose: bool = False
) -> Dict:
    """Main function to generate educational images"""
    if verbose:
        logger.info(f"Generating educational image for prompt: {prompt}")

    generator = ImageGenerator(verbose=verbose, project_id=project_id)
    
    logger.info(f"Prompt: {prompt}")
    # Safety check
    if not generator.check_safety(prompt):
        raise ValueError("Prompt failed safety check")

    logger.info(f"Passed Safety Check")

    # Enhance prompt
    enhanced_prompt = generator.enhance_prompt(prompt, subject, grade_level)

    logger.info(f"Enhanced Prompt: {enhanced_prompt}")

    # Generate image
    image_url = generator.generate_image(enhanced_prompt, lang)

    output = ImageGenerationOutput(
        image_url=image_url,
        prompt_used=enhanced_prompt,
        safety_status="PASSED",
        educational_context={
            "subject": subject,
            "grade_level": grade_level,
            "original_prompt": prompt,
            "language": lang
        }
    )
    
    return dict(output)
































