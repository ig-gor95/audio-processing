import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydub import AudioSegment
from pathlib import Path
import soundfile as sf
import os
import uuid

from core.audio_loader import AudioLoader
from utils.log_utils import setup_logger

logger = setup_logger(__name__)
app = FastAPI(title="Audio Analysis API")

# Configuration
AUDIO_UPLOAD_DIR = Path("data/input/temp")
AUDIO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize AudioLoader with default config
audio_loader = AudioLoader(sample_rate=16000, mono=True)
logger.info("AudioLoader initialized")


def export_audio(audio_data: np.ndarray, sample_rate: int, output_path: Path):
    """Export numpy audio array to wav using pydub."""
    if audio_data.ndim == 2:
        # shape (samples, channels)
        if audio_data.shape[1] != 2 and audio_data.shape[0] == 2:
            audio_data = audio_data.T  # transpose if needed to (samples, channels)

        audio_int16 = (audio_data * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=2,
        )
    else:
        # Mono audio
        audio_int16 = (audio_data * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1,
        )
    audio_segment.export(str(output_path), format="wav")


@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    logger.info(f"Received upload request for file: {file.filename}")

    try:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in audio_loader.supported_formats:
            raise HTTPException(status_code=400, detail=f"Unsupported audio format: {file_ext}")

        unique_filename = f"{uuid.uuid4()}{file_ext}"
        temp_path = AUDIO_UPLOAD_DIR / unique_filename

        # Save uploaded file
        logger.debug(f"Saving file to: {temp_path}")
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # Load audio
        audio_data, sample_rate = audio_loader.load_audio(str(temp_path))

        # Normalize audio
        processed_audio = audio_loader.normalize_audio(audio_data, sample_rate)
        processed_audio = audio_loader.reduce_noise(audio_data, sample_rate)

        # Remove silence
        # processed_audio = audio_loader.remove_silence(normalized_audio, sample_rate)

        # Export processed audio to wav
        processed_filename = f"processed_{unique_filename}"
        processed_path = AUDIO_UPLOAD_DIR / Path(processed_filename).with_suffix(".wav")

        export_audio(processed_audio, sample_rate, processed_path)

        # Cleanup original upload
        temp_path.unlink()

        logger.info(f"Successfully processed file: {file.filename}")
        return JSONResponse({
            "status": "success",
            "original_file": file.filename,
            "processed_file": str(processed_path),
            "duration": audio_loader.get_audio_duration(processed_audio, sample_rate),
            "sample_rate": sample_rate,
        })

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/download-audio/{filename}")
async def download_audio(filename: str):
    file_path = AUDIO_UPLOAD_DIR / filename

    if not file_path.exists():
        logger.error(f"File not found: {filename}")
        raise HTTPException(status_code=404, detail="File not found")

    logger.debug(f"Serving file: {file_path}")
    return FileResponse(file_path)


@app.post("/batch-process/")
async def batch_process(directory: str):
    logger.info(f"Batch processing request for directory: {directory}")

    try:
        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            raise HTTPException(status_code=400, detail="Directory not found")

        results = []
        loaded_files = audio_loader.batch_load(directory)
        logger.debug(f"Found {len(loaded_files)} files to process")

        for file_path, audio_data, sample_rate in loaded_files:
            try:
                # Normalize and remove silence
                normalized_audio = audio_loader.normalize_audio(audio_data, sample_rate)
                processed_audio = audio_loader.remove_silence(normalized_audio, sample_rate)

                output_path = Path(file_path).with_name(f"processed_{Path(file_path).name}").with_suffix(".wav")

                export_audio(processed_audio, sample_rate, output_path)

                results.append({
                    "original_file": file_path,
                    "processed_file": str(output_path),
                    "duration": audio_loader.get_audio_duration(processed_audio, sample_rate),
                })

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue

        logger.info(f"Batch processing completed. {len(results)} files processed successfully")
        return JSONResponse({"status": "success", "processed_files": results})

    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
