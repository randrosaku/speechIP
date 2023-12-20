import torch
from huggingsound import SpeechRecognitionModel

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
model = SpeechRecognitionModel(
    "jonatasgrosman/wav2vec2-large-xlsr-53-english", device=device
)

references = [
    {
        "path": "./audio/h5.wav",
        "transcription": "tacos al pastor are my favorite",
    },
    {
        "path": "./audio/h6.wav",
        "transcription": "a zestful food is the hot cross bun",
    },
]

evaluation = model.evaluate(references, inference_batch_size=batch_size)

print(evaluation)
