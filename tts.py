import argparse
import librosa
import numpy
import os
from scipy.io.wavfile import write
import sys
import torch
import tps
import yaml
from yaml.loader import SafeLoader

path_to_tacotron = "/home/max/TTS/pycharm-sova/sova-tts-engine"
path_to_waveglow = "/home/max/TTS/pycharm-sova/waveglow"
path_to_waveglow_tacotron2 = "/home/max/TTS/pycharm-sova/waveglow/tacotron2/"

#sys.path.insert(0, path_to_waveglow_tacotron2)
sys.path.insert(0, path_to_waveglow)
#from denoiser import Denoiser
MAX_WAV_VALUE = 32768.0  # value from mel2samp.py in waveglow

sys.path.insert(0, path_to_tacotron)
from hparams import create_hparams
from model import load_model
from utils.data_utils import TextMelLoader


def get_text(text, mask_stress, mask_phonemes, text_handler):
    preprocessed_text = text_handler.process_text(
        text, tps.cleaners.light_punctuation_cleaners, None, False,
        mask_stress=mask_stress, mask_phonemes=mask_phonemes
    )
    preprocessed_text = text_handler.check_eos(preprocessed_text)
    text_vector = text_handler.text2vec(preprocessed_text)

    text_tensor = torch.IntTensor(text_vector)
    return text_tensor


def get_text_and_names_from_csv(csv_path):
    names = []
    texts = []
    return names, texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--config_path", type=str, default="tts_config.yml",
                        required=False, help="Path to the configuration file")
    parser.add_argument("-t", "--text_to_speech", type=str, default="",
                        required=False, help="Text to synthesize")
    parser.add_argument("-n", "--file_name", type=str, default="synthesis.wav",
                        required=False, help="Name of the result .wav file")
    parser.add_argument("-c", "--csv_to_speech", type=str, default="",
                        required=False, help="Path to .csv file with text for synthesis")
    args = parser.parse_args()

    config_path = args.config_path
    console_text = args.text_to_speech
    console_name = args.file_name
    csv_path = args.csv_to_speech

    names = []
    texts = []
    if csv_path != "":
        names, texts = get_text_and_names_from_csv(csv_path)

    if console_text != "":
        names.append(console_name)
        texts.append(console_text)

    assert os.path.isfile(config_path)

    with open(config_path) as config_file:
        data = yaml.load(config_file, Loader=SafeLoader)
        hparams_path = data['hparams']
        checkpoint_path = data['checkpoint']
        use_gst = data['use_gst']
        if use_gst:
            emotion_wav = data['emotion_wav']
        output_folder = data['output_folder']

        waveglow_path = data['waveglow_path']
        is_fp16 = data['is_fp16']
        #denoiser_strength = data['denoiser_strength']
        sigma = data['sigma']
        sampling_rate = data['sampling_rate']

        print("Config file loaded successfully")

    # Waveglow initialization
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()

    if is_fp16:
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

    print("Waveglow initialized")

    # Tacotron2 initialization
    assert os.path.isfile(hparams_path)
    hparams = create_hparams(hparams_path)
    hparams.path = hparams_path

    model = load_model(hparams, False)

    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))

    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])

    model.cuda().eval()

    charset = 'en'
    text_handler = tps.Handler(charset)

    if use_gst:
        test_csv = "/home/max/TTS/pycharm-sova/inference_gst_test.csv"
        text_loader = TextMelLoader(text_handler, test_csv, hparams)
        ref_mel = torch.unsqueeze(text_loader.get_mel(emotion_wav), 0).cuda()

    mask_stress = False
    mask_phonemes = True

    print("Tacotron2 initialized")

    # Work cycle
    with torch.no_grad():
        for input_sentence, input_name in zip(texts, names):
            processed_sentence = get_text(input_sentence, mask_stress, mask_phonemes, text_handler)
            processed_sentence = torch.unsqueeze(processed_sentence, 0)
            processed_sentence = processed_sentence.cuda()
            if use_gst:
                mel_outputs, mel_outputs_postnet, gates, alignments = model.inference(processed_sentence,
                                                                                      reference_mel=ref_mel)
            else:
                mel_outputs, mel_outputs_postnet, gates, alignments = model.inference(processed_sentence)
            #mel_save_path = os.path.join(output_folder, f'{input_name}.pt')
            #torch.save(mel_outputs_postnet[0], mel_save_path)

            #file_name = os.path.splitext(os.path.basename(mel_save_path))[0]
            #mel = torch.load(mel_save_path)
            mel = mel_outputs_postnet[0]
            mel = torch.autograd.Variable(mel.cuda())
            mel = torch.unsqueeze(mel, 0)
            mel = mel.half() if is_fp16 else mel
            audio = waveglow.infer(mel, sigma=sigma)
            audio = audio * MAX_WAV_VALUE
            audio = audio.squeeze()
            audio = audio.cpu().numpy()
            audio = audio.astype('int16')
            audio_path = os.path.join(
                output_folder, "{}_synthesis.wav".format(input_name))
            write(audio_path, sampling_rate, audio)
            print(audio_path)
