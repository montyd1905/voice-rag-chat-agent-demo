import whisper
import logging
from typing import Optional
import io
import numpy as np
from config import settings

logger = logging.getLogger(__name__)

# global model instance
_stt_model = None
_current_model_name = None


def get_stt_model(model_name: Optional[str] = None):
    """Get or load the STT model"""
    global _stt_model, _current_model_name
    
    # Use model from settings if not specified
    if model_name is None:
        model_name = settings.WHISPER_MODEL
    
    # Reload model if name changed
    if _stt_model is None or _current_model_name != model_name:
        logger.info(f"Loading STT model: {model_name}")
        _stt_model = whisper.load_model(model_name)
        _current_model_name = model_name
    return _stt_model


class STTService:
    """Service for Speech-to-Text using Whisper with improved audio preprocessing"""
    
    @staticmethod
    def transcribe(audio_data: bytes, model_name: Optional[str] = None) -> Optional[str]:
        """Transcribe audio to text with improved preprocessing"""
        try:
            model = get_stt_model(model_name)
            
            # Load and preprocess audio
            import tempfile
            import os
            import soundfile as sf
            
            # Save audio to temporary file first
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            try:
                # Load audio with soundfile for better format handling
                audio_array, sample_rate = sf.read(tmp_file_path)
                
                # Normalize audio to prevent clipping and improve recognition
                if len(audio_array.shape) > 1:
                    # Convert stereo to mono by averaging channels
                    audio_array = np.mean(audio_array, axis=1)
                
                # Normalize to [-1, 1] range if needed
                max_val = np.abs(audio_array).max()
                if max_val > 1.0:
                    audio_array = audio_array / max_val
                
                # Whisper expects 16kHz sample rate, resample if needed
                if sample_rate != 16000:
                    from scipy import signal
                    num_samples = int(len(audio_array) * 16000 / sample_rate)
                    audio_array = signal.resample(audio_array, num_samples)
                    sample_rate = 16000
                
                # Save preprocessed audio
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as processed_file:
                    processed_path = processed_file.name
                
                sf.write(processed_path, audio_array, sample_rate)
                
                # Transcribe with optimized settings
                result = model.transcribe(
                    processed_path,
                    language="en",
                    task="transcribe",
                    fp16=False,  # Use fp32 for better accuracy
                    verbose=False
                )
                text = result["text"].strip()
                
                # Clean up
                os.unlink(processed_path)
                
                return text
            finally:
                # Always clean up original temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
            
        except ImportError as e:
            # Fallback if soundfile/scipy not available
            logger.warning(f"Audio preprocessing libraries not available: {e}, using basic transcription")
            try:
                model = get_stt_model(model_name)
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_data)
                    tmp_file_path = tmp_file.name
                
                result = model.transcribe(tmp_file_path, language="en", fp16=False)
                text = result["text"].strip()
                os.unlink(tmp_file_path)
                return text
            except Exception as fallback_error:
                logger.error(f"Fallback STT transcription failed: {fallback_error}")
                return None
        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            return None

