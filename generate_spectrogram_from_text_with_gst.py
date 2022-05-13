import sys
import os
import argparse
import torch

import tps

path_to_tacotron = "/home/max/TTS/pycharm-sova/sova-tts-engine"
sys.path.insert(0, path_to_tacotron)

from model import load_model
from hparams import create_hparams

from utils.data_utils import TextMelLoader
from scipy.io.wavfile import read
import librosa


def get_text(text, mask_stress, mask_phonemes, text_handler):
    preprocessed_text = text_handler.process_text(
        text, tps.cleaners.light_punctuation_cleaners, None, False,
        mask_stress=mask_stress, mask_phonemes=mask_phonemes
    )
    preprocessed_text = text_handler.check_eos(preprocessed_text)
    text_vector = text_handler.text2vec(preprocessed_text)

    text_tensor = torch.IntTensor(text_vector)
    return text_tensor


if __name__ == "__main__":
    checkpoint_path = "/home/max/TTS/pycharm-sova/sova-tts-engine/output_dir/checkpoint_49000"
    #audio_path = "/home/max/Downloads/Emotional Speech Dataset (ESD)/0019/Angry/test/0019_000371.wav"
    #audio_path = "/home/max/Downloads/Emotional Speech Dataset (ESD)/0019/Happy/test/0019_000721.wav"
    #audio_path = "/home/max/Downloads/Emotional Speech Dataset (ESD)/0019/Neutral/test/0019_000021.wav"
    audio_path = "/home/max/Downloads/Emotional Speech Dataset (ESD)/0019/Sad/test/0019_001071.wav"
    save_dir = "/home/max/TTS/pycharm-sova/sova-tts-engine/output_dir/"
    test_csv = "/home/max/TTS/pycharm-sova/inference_gst_test.csv"

    input_sentences = ["Just sample text for testing"]
    input_names = ['Molly']

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams_path", type=str, default="/home/max/TTS/pycharm-sova/sova-tts-engine/data/hparams.yaml",
                        required=False, help="hparams path")
    args = parser.parse_args()

    hparams = create_hparams(args.hparams_path)
    hparams.path = args.hparams_path

    model = load_model(hparams, False)

    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))

    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])

    model.cuda().eval()

    charset = 'en'
    text_handler = tps.Handler(charset)

    text_loader = TextMelLoader(text_handler, test_csv, hparams)
    y, sr = librosa.load(audio_path)
    mel = torch.tensor(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=hparams.n_mel_channels))
    mel = torch.unsqueeze(text_loader.get_mel(audio_path), 0).cuda()
    # mel = torch.unsqueeze(mel, 0).cuda()
    print(mel.shape)

    mask_stress = False
    mask_phonemes = True

    with torch.no_grad():
        for input_sentence, input_name in zip(input_sentences, input_names):
            processed_sentence = get_text(input_sentence, mask_stress, mask_phonemes, text_handler)
            processed_sentence = torch.unsqueeze(processed_sentence, 0)
            processed_sentence = processed_sentence.cuda()
            mel_outputs, mel_outputs_postnet, gates, alignments = model.inference(processed_sentence, reference_mel=mel)
            mel_save_path = os.path.join(save_dir, f'{input_name}.pt')
            print(mel_save_path)
            torch.save(mel_outputs_postnet[0], mel_save_path)
