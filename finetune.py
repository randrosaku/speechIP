import torch
from huggingsound import (
    TrainingArguments,
    ModelArguments,
    SpeechRecognitionModel,
    TokenSet,
)
from datasets import load_dataset

train_dataset = load_dataset(
    "mozilla-foundation/common_voice_11_0", "lt", split="train[:1000]"
)
train_data = train_dataset.rename_column("sentence", "transcription")

eval_dataset = load_dataset(
    "mozilla-foundation/common_voice_11_0", "lt", split="validation[:200]"
)
eval_data = eval_dataset.rename_column("sentence", "transcription")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SpeechRecognitionModel("facebook/wav2vec2-large-xlsr-53", device="cpu")
output_dir = "./output/"

# first of all, you need to define your model's token set
tokens = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "'",
    "ą",
    "č",
    "ę",
    "ė",
    "į",
    "š",
    "ų",
    "ū",
    "ž",
]
token_set = TokenSet(tokens)

# the lines below will load the training and model arguments objects,
# you can check the source code (huggingsound.trainer.TrainingArguments and huggingsound.trainer.ModelArguments) to see all the available arguments
# training_args = TrainingArguments(
#     learning_rate=3e-4,
#     max_steps=1000,
#     eval_steps=200,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
# )
# model_args = ModelArguments(
#     activation_dropout=0.1,
#     hidden_dropout=0.1,
# )

# test1
training_args = TrainingArguments(
    learning_rate=5e-4,
    max_steps=3000,
    eval_steps=500,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
)
model_args = ModelArguments(
    activation_dropout=0.2,
    hidden_dropout=0.2,
)

train = [
    {"path": path, "transcription": transcription}
    for path, transcription in zip(train_data["path"], train_data["transcription"])
]

eval = [
    {"path": path, "transcription": transcription}
    for path, transcription in zip(train_data["path"], train_data["transcription"])
]


model.finetune(
    output_dir,
    train_data=train,
    eval_data=eval,
    token_set=token_set,
    training_args=training_args,
    model_args=model_args,
)
