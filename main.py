from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import torch
import soundfile as sf
import io

app = FastAPI()

# Load model once
model, _ = torch.hub.load(
    "snakers4/silero-models",
    "silero_tts",
    language="en",
    speaker="v3_en"
)
@app.get("/")
def home():
    return {"message":"welcome to world tts a tts service api vist /docs for more."}
@app.get("/tts")
def tts(text: str):

    audio = model.apply_tts(
        text=text,
        speaker="en_3",
        sample_rate=48000
    )

    buffer = io.BytesIO()
    sf.write(buffer, audio, 48000, format="WAV")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav"
    )
