import torch
from huggingsound import SpeechRecognitionModel
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
model = SpeechRecognitionModel(
    "jonatasgrosman/wav2vec2-large-xlsr-53-english", device=device
)

LANG_ID = "lt"
SAMPLES = 10

test_dataset = load_dataset(
    "mozilla-foundation/common_voice_11_0", LANG_ID, split=f"test[:{SAMPLES}]"
)

transcriptions = model.transcribe(test_dataset["path"], batch_size=batch_size)
t = [d["transcription"] for d in transcriptions]

print("ACTUAL")
print(test_dataset["sentence"])
print()
print("PREDICTED")
print(t)
