import whisper
import logging
from typing import Optional
import tempfile
import os
from config import settings

logger = logging.getLogger(__name__)

# global model instance
_stt_model = None
_current_model_name = None


def get_stt_model(model_name: Optional[str] = None):
    """Get or load the STT model"""
    global _stt_model, _current_model_name
    
    # use model from settings if not specified
    if model_name is None:
        model_name = settings.WHISPER_MODEL
    
    # reload model if name changed
    if _stt_model is None or _current_model_name != model_name:
        logger.info(f"Loading STT model: {model_name}")
        _stt_model = whisper.load_model(model_name)
        _current_model_name = model_name
    return _stt_model


class STTService:
    """Service for Speech-to-Text using Whisper - handles format detection and conversion automatically"""
    
    @staticmethod
    def transcribe(audio_data: bytes, model_name: Optional[str] = None) -> Optional[str]:
        """Transcribe audio to text - Whisper handles format detection and conversion automatically"""
        model = get_stt_model(model_name)
        audio_path = None
        
        try:
            # save raw audio data to temporary file
            # Whisper can automatically detect and handle multiple formats
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
                tmp_file.write(audio_data)
                audio_path = tmp_file.name
            
            # Whisper automatically handles:
            # - Format detection (WebM, Opus, MP3, WAV, etc.)
            # - Audio conversion and resampling
            # - Channel conversion (stereo to mono)
            # - Normalization
            result = model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                fp16=False,  # Use fp32 for better accuracy
                verbose=False
            )
            
            text = result["text"].strip()
            return text
                
        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            return None
        finally:
            # clean up temp file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass

