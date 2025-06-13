import whisper
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import argparse
from art.estimators.classification.pytorch import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent

# --- Fixed Configuration ---
# This value is the chunk size for the Whisper model
MAX_AUDIO_SECONDS = 30

# The ART Wrapper for OpenAI's Whisper Model
class PyTorchWhisper(PyTorchClassifier):
    """
    An ART wrapper for OpenAI's Whisper model.
    """
    def __init__(self, model: whisper.model.Whisper):
        tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language="en")
        super().__init__(
            model=model,
            loss=nn.CrossEntropyLoss(),
            input_shape=(whisper.audio.N_SAMPLES,),
            nb_classes=model.dims.n_vocab,
            optimizer=None,
            channels_first=False,
            clip_values=(-1.0, 1.0),
            device_type=str(model.device)
        )
        self.tokenizer = tokenizer

    def predict(self, x: np.ndarray, batch_size: int = 1, **kwargs):
        self.model.eval()
        dummy_logits = np.zeros((x.shape[0], self.nb_classes), dtype=np.float32)
        return dummy_logits

    def loss_gradient(self, x, y):
        self.model.train()

        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).to(self.device).float()
        else:
            x_tensor = x.to(self.device)

        # Create a new leaf tensor for the current computation
        x_tensor = x_tensor.clone().detach().requires_grad_(True)

        if isinstance(y, np.ndarray):
            y_tensor = torch.from_numpy(y).to(self.device).long()
        else:
            y_tensor = y.to(self.device).long()

        mel = whisper.audio.log_mel_spectrogram(x_tensor)
        encoder_output = self.model.encoder(mel)
        sot_token = self.tokenizer.sot
        decoder_input = torch.cat([torch.tensor([sot_token]), y_tensor.squeeze()])[:-1].unsqueeze(0)
        logits = self.model.decoder(decoder_input, encoder_output)
        loss = self._loss(logits.view(-1, self.nb_classes), y_tensor.view(-1))
        
        self.model.zero_grad()
        loss.backward()
        gradient = x_tensor.grad
        return gradient


# Helper Functions

def load_and_prepare_audio(audio_path, max_len_seconds, sample_rate=whisper.audio.SAMPLE_RATE):
    try:
        audio_data, sr = sf.read(audio_path, dtype='float32')
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None
    if sr != sample_rate:
        raise ValueError(f"Audio file needs sample rate {sample_rate}")
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data /= max_val
    target_length = max_len_seconds * sample_rate
    if len(audio_data) < target_length:
        padded_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')
    else:
        padded_data = audio_data[:target_length]
    return padded_data.astype(np.float32)

def save_audio(audio_data, file_path, sample_rate=whisper.audio.SAMPLE_RATE):
    sf.write(file_path, audio_data, sample_rate)
    print(f"Saved audio to {file_path}")


# PART 3: Main Execution Block
def main(
    input: str,
    output: str,
    target_phrase: str,
    whisper_model_name: str
    ):
    print(f"Loading Whisper model '{whisper_model_name}'...")
    whisper_model = whisper.load_model(whisper_model_name, device="cpu")

    print("Wrapping model with ART estimator...")
    estimator = PyTorchWhisper(model=whisper_model)

    print(f"Loading original audio from '{input}'...")
    original_audio = load_and_prepare_audio(input, MAX_AUDIO_SECONDS)
    original_audio_batch = np.expand_dims(original_audio, axis=0)

    print("\n--- Transcribing Original Audio ---")
    original_transcription = whisper.transcribe(whisper_model, input, fp16=False)["text"]

    print(f"Original Transcription: '{original_transcription.strip()}'")


    print(f"\n--- Preparing Adversarial Attack ---")
    print(f"Target Phrase: '{target_phrase}'")
    target_tokens = estimator.tokenizer.encode(target_phrase)
    target_tokens_batch = np.expand_dims(np.array(target_tokens, dtype=np.int32), axis=0)

    attack = ProjectedGradientDescent(
        estimator=estimator,
        eps=0.1,
        eps_step=0.005,
        max_iter=50,
        targeted=True,
        verbose=True
    )

    print("\n--- Generating Adversarial Audio (this may take a while)... ---")
    adversarial_audio_batch = attack.generate(x=original_audio_batch, y=target_tokens_batch)
    adversarial_audio = adversarial_audio_batch.squeeze()

    print("\n--- Saving and Verifying Attack ---")
    # ART's output is a tensor, convert it back to numpy for saving
    save_audio(adversarial_audio, output)
    try:
        adversarial_transcription = whisper.transcribe(whisper_model, output, fp16=False)["text"]
        print(f"Adversarial Transcription: '{adversarial_transcription.strip()}'")
    except Exception as e:
        print(f"Could not transcribe adversarial audio: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- Configuration Arguments ---
    parser.add_argument(
        '--input', type=str, default="my_original_command.wav",
    )
    parser.add_argument(
        '--output', type=str, default="adversarial_output.wav",
    )
    parser.add_argument(
        '--target-phrase', type=str, default="Hey Sir open the door",
        help="The target phrase for the attack."
    )
    parser.add_argument(
        '--whisper-model-name',
        type=str,
        default="base.en",
        choices=['tiny.en', 'small.en', 'medium.en', 'base.en'],
    )

    args = parser.parse_args()

    main(
        input=args.input,
        output=args.output,
        target_phrase=args.target_phrase,
        whisper_model_name=args.whisper_model_name
    )