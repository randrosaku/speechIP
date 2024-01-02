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
test_dataset = test_dataset.rename_column("sentence", "transcription")

evaluation = model.evaluate(test_dataset, inference_batch_size=batch_size)

print(evaluation)
