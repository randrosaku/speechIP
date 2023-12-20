import torch
from huggingsound import (
    TrainingArguments,
    ModelArguments,
    SpeechRecognitionModel,
    TokenSet,
)
from datasets import load_dataset

librispeech_dataset = load_dataset("librispeech_asr", "clean", split="train.100")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SpeechRecognitionModel("facebook/wav2vec2-large-xlsr-53", device=device)
output_dir = "./output_origin/"

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
]
token_set = TokenSet(tokens)

# the lines below will load the training and model arguments objects,
# you can check the source code (huggingsound.trainer.TrainingArguments and huggingsound.trainer.ModelArguments) to see all the available arguments
training_args = TrainingArguments(
    learning_rate=3e-4,
    max_steps=1000,
    eval_steps=200,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
)
model_args = ModelArguments(
    activation_dropout=0.1,
    hidden_dropout=0.1,
)

train_data = [
    {"path": audio_path, "transcription": transcription}
    for audio_path, transcription in zip(
        librispeech_dataset["file"], librispeech_dataset["text"]
    )
]

# Load validation data (you can choose another split like 'validation' or 'test')
eval_dataset = load_dataset("librispeech_asr", "clean", split="validation")
eval_data = [
    {"path": audio_path, "transcription": transcription}
    for audio_path, transcription in zip(eval_dataset["file"], eval_dataset["text"])
]

# # define your train/eval data
# train_data = [
#     {
#         "path": "./audio/h1.wav",
#         "transcription": "the stale smell of old beer lingers",
#     },
#     {
#         "path": "./audio/h2.wav",
#         "transcription": "it takes heat to bring out the odor",
#     },
#     {
#         "path": "./audio/h3.wav",
#         "transcription": "a cold dip restores Health in zest",
#     },
#     {
#         "path": "./audio/h4.wav",
#         "transcription": "a salt pickle taste fine with ham",
#     },
# ]
# eval_data = [
#     {
#         "path": "./audio/h5.wav",
#         "transcription": "tacos al pastor are my favorite",
#     },
#     {
#         "path": "./audio/h6.wav",
#         "transcription": "a zestful food is the hot cross bun",
#     },
# ]


# and finally, fine-tune your model
model.finetune(
    output_dir,
    train_data=train_data,
    eval_data=eval_data,  # the eval_data is optional
    token_set=token_set,
    training_args=training_args,
    model_args=model_args,
)
