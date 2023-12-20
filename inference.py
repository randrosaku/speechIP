import torch
from huggingsound import SpeechRecognitionModel

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
model = SpeechRecognitionModel(
    "jonatasgrosman/wav2vec2-large-xlsr-53-english", device=device
)
audio_paths = ["./audio/h1.wav", "./audio/h2.wav"]

transcriptions = model.transcribe(audio_paths, batch_size=batch_size)

print(transcriptions)
