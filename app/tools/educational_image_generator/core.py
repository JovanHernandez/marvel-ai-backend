from app.services.logger import setup_logger
from app.tools.educational_image_generator.tools import generate_educational_image
import os

logger = setup_logger(__name__)

def executor(prompt: str, subject: str, grade_level: str, lang: str = "en", verbose: bool = False):
    """
    Main executor function for the educational image generator tool
    
    Args:
        prompt (str): The image generation prompt
        subject (str): The subject area
        grade_level (str): The target grade level
        lang (str, optional): Language code. Defaults to "en"
        verbose (bool, optional): Enable verbose logging. Defaults to False
    
    Returns:
        dict: Generated image data including URL and metadata
    """
    try:
        if verbose:
            logger.info(f"Starting educational image generation for prompt: {prompt}")
        
        project_id = os.getenv('PROJECT_ID')
        if not project_id:
            raise ValueError("PROJECT_ID environment variable is not set")
        
        result = generate_educational_image(
            prompt=prompt,
            subject=subject,
            grade_level=grade_level,
            project_id=project_id,
            lang=lang,
            verbose=verbose
        )
        
        if verbose:
            logger.info("Successfully generated educational image")
            
        return result
        
    except Exception as e:
        logger.error(f"Error in Educational Image Generator: {str(e)}")
        raise

