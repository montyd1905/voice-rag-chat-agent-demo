import logging
from typing import Optional
import io
import numpy as np

logger = logging.getLogger(__name__)

# global model instance
_tts_model = None
_tts_vocoder = None


def get_tts_model():
    """Get or load the TTS model"""
    global _tts_model, _tts_vocoder
    if _tts_model is None:
        try:
            from TTS.api import TTS
            logger.info("Loading TTS model")
            _tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            _tts_model = None
    return _tts_model


class TTSService:
    """Service for Text-to-Speech using Coqui TTS"""
    
    @staticmethod
    def synthesize(text: str) -> Optional[bytes]:
        """Convert text to speech audio"""
        try:
            model = get_tts_model()
            if model is None:
                logger.error("TTS model not available")
                return None
            
            # generate audio
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file_path = tmp_file.name
            
            model.tts_to_file(text=text, file_path=tmp_file_path)
            
            # read audio file
            with open(tmp_file_path, 'rb') as f:
                audio_data = f.read()
            
            # clean up and return data
            os.unlink(tmp_file_path)
            
            return audio_data
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None

