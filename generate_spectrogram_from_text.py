import sys
import os

import argparse
import torch
import tps

path_to_tacotron = "/home/max/TTS/pycharm-sova/sova-tts-engine"
sys.path.insert(0, path_to_tacotron)

from model import load_model
from hparams import create_hparams


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
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams_path", type=str, default="/home/max/TTS/pycharm-sova/sova-tts-engine/data/hparams_without_gst.yaml",
                        required=False, help="hparams path")
    args = parser.parse_args()

    hparams = create_hparams(args.hparams_path)
    hparams.path = args.hparams_path

    model = load_model(hparams, False)
    checkpoint_path = "/home/max/TTS/pycharm-sova/sova-tts-engine/output_dir/trained_without_gst"
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))

    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])

    model.cuda().eval()

    charset = 'en'
    text_handler = tps.Handler(charset)

    input_sentences = ["Just sample text for testing"]
    input_names = ['Molly']

    mask_stress = False
    mask_phonemes = True
    save_dir = "/home/max/TTS/pycharm-sova/sova-tts-engine/output_dir/"

    with torch.no_grad():
        for input_sentence, input_name in zip(input_sentences, input_names):
            processed_sentence = get_text(input_sentence, mask_stress, mask_phonemes, text_handler)
            processed_sentence = torch.unsqueeze(processed_sentence, 0)
            processed_sentence = processed_sentence.cuda()
            mel_outputs, mel_outputs_postnet, gates, alignments = model.inference(processed_sentence)
            mel_save_path = os.path.join(save_dir, f'{input_name}.pt')
            print(mel_save_path)
            torch.save(mel_outputs_postnet[0], mel_save_path)
            #print(mel_outputs_postnet[0])
