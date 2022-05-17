import argparse
import os
from scipy.io.wavfile import write
import torch
import yaml
from yaml.loader import SafeLoader

import synthesizer


def get_text_and_names_from_csv(csv_path):
    names = []
    texts = []
    for line in open(csv_path, 'r'):
        line = line.split('|')
        texts.append(line[1])
        names.append(line[0])
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
    parser.add_argument("-s", "--use_ssml", action='store_true', default=False,
                        required=False, help="Enable SSML parsing")
    args = parser.parse_args()

    config_path = args.config_path
    console_text = args.text_to_speech
    console_name = args.file_name
    csv_path = args.csv_to_speech
    use_ssml = args.use_ssml

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
        sigma = data['sigma']
        sampling_rate = data['sampling_rate']

        print("Config file loaded successfully")

    mask_stress = False
    mask_phonemes = True

    synthesizer = synthesizer.Synthesizer(waveglow_path=waveglow_path,
                                          is_fp16=is_fp16,
                                          sampling_rate=sampling_rate,
                                          sigma=sigma,
                                          hparams_path=hparams_path,
                                          checkpoint_path=checkpoint_path,
                                          use_gst=use_gst,
                                          emotion_wav=emotion_wav,
                                          mask_stress=mask_stress,
                                          mask_phonemes=mask_phonemes)
    # Work cycle
    with torch.no_grad():
        for input_sentence, input_name in zip(texts, names):
            audio = synthesizer.synthesize(input_sentence, input_name)
            audio_path = os.path.join(output_folder, "{}".format(input_name))
            write(audio_path, sampling_rate, audio)
            print(audio_path)
