from typing import Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from app.services.logger import setup_logger
import os
import base64
from io import BytesIO
from langchain_core.messages import HumanMessage

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
    ):
        self.verbose = verbose
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

    def generate_image(self, enhanced_prompt: str) -> str:
        """
        Placeholder for image generation - returns a hardcoded URL
        
        Args:
            enhanced_prompt (str): The enhanced prompt for image generation
        
        Returns:
            str: URL of the generated image
        """
        if self.verbose:
            logger.info("Returning placeholder image URL")
        
        # Return a placeholder URL - replace with your desired default image URL
        return "https://placeholder.com/educational-image.png"

def generate_educational_image(prompt: str, subject: str, grade_level: str, lang: str = "en", verbose: bool = False) -> Dict:
    """Main function to generate educational images"""
    if verbose:
        logger.info(f"Generating educational image for prompt: {prompt}")

    generator = ImageGenerator(verbose=verbose)
    
    logger.info(f"Prompt: {prompt}")
    # Safety check
    if not generator.check_safety(prompt):
        raise ValueError("Prompt failed safety check")

    logger.info(f"Passed Safety Check")

    # Enhance prompt
    enhanced_prompt = generator.enhance_prompt(prompt, subject, grade_level)

    logger.info(f"Enhanced Prompt: {enhanced_prompt}")

    # Generate image
    image_url = generator.generate_image(enhanced_prompt)

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











