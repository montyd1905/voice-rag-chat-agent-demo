import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import io
import logging

logger = logging.getLogger(__name__)


class OCRService:
    """Service for Optical Character Recognition"""
    
    @staticmethod
    def extract_text_from_image(image_data: bytes) -> str:
        """Extract text from image bytes"""
        try:
            image = Image.open(io.BytesIO(image_data))
            text = pytesseract.image_to_string(image, lang='eng')
            return text
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise
    
    @staticmethod
    def extract_text_from_pdf(pdf_data: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            # save PDF to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_data)
                tmp_file_path = tmp_file.name
            
            # convert PDF to images
            images = convert_from_path(tmp_file_path)
            
            # extract text from each page
            text_parts = []
            for image in images:
                text = pytesseract.image_to_string(image, lang='eng')
                text_parts.append(text)
            
            # clean up
            import os
            os.unlink(tmp_file_path)
            
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"PDF OCR extraction failed: {e}")
            raise
    
    @staticmethod
    def extract_text(file_data: bytes, file_type: str) -> str:
        """Extract text from file based on type"""
        if file_type.lower() in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
            return OCRService.extract_text_from_image(file_data)
        elif file_type.lower() == 'pdf':
            return OCRService.extract_text_from_pdf(file_data)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

