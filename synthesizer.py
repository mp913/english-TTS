import numpy as np
import os
import sys
import torch
import tps


path_to_tacotron = "/home/max/TTS/pycharm-sova/sova-tts-engine"
path_to_waveglow = "/home/max/TTS/pycharm-sova/waveglow"

from tps.modules.ssml.parser import parse_ssml_text
from ssml_controller import SSMLException, ssml_factory

sys.path.insert(0, path_to_waveglow)
MAX_WAV_VALUE = 32768.0  # value from mel2samp.py in waveglow

sys.path.insert(0, path_to_tacotron)
from hparams import create_hparams
from model import load_model
from utils.data_utils import TextMelLoader


class Synthesizer:
    def __init__(self, waveglow_path, is_fp16, sampling_rate, sigma, hparams_path, checkpoint_path, use_gst, emotion_wav, mask_stress,
                 mask_phonemes):
        # Waveglow initialization
        self.waveglow = torch.load(waveglow_path)['model']
        self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
        self.waveglow.cuda().eval()

        self.is_fp16 = is_fp16
        self.sampling_rate = sampling_rate
        self.sigma = sigma

        if is_fp16:
            from apex import amp
            self.waveglow, _ = amp.initialize(self.waveglow, [], opt_level="O3")

        print("Waveglow initialized")

        # Tacotron2 initialization
        assert os.path.isfile(hparams_path)
        hparams = create_hparams(hparams_path)
        hparams.path = hparams_path

        self.model = load_model(hparams, False)

        assert os.path.isfile(checkpoint_path)
        print("Loading checkpoint '{}'".format(checkpoint_path))

        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint_dict["state_dict"])

        self.model.cuda().eval()

        self.charset = 'en'
        self.text_handler = tps.Handler(self.charset)

        self.use_gst = use_gst
        if self.use_gst:
            test_csv = "/home/max/TTS/pycharm-sova/inference_gst_test.csv"
            self.text_loader = TextMelLoader(self.text_handler, test_csv, hparams)
            self.ref_mel = torch.unsqueeze(self.text_loader.get_mel(emotion_wav), 0).cuda()

        self.mask_stress = mask_stress
        self.mask_phonemes = mask_phonemes

        print("Tacotron2 initialized")

    @staticmethod
    def get_text(text, mask_stress, mask_phonemes, text_handler):
        preprocessed_text = text_handler.process_text(
            text, tps.cleaners.light_punctuation_cleaners, None, False,
            mask_stress=mask_stress, mask_phonemes=mask_phonemes
        )
        preprocessed_text = text_handler.check_eos(preprocessed_text)
        text_vector = text_handler.text2vec(preprocessed_text)

        text_tensor = torch.IntTensor(text_vector)
        return text_tensor

    def synthesize(self, input_sentence):
        processed_sentence = self.get_text(input_sentence, self.mask_stress, self.mask_phonemes, self.text_handler)
        processed_sentence = torch.unsqueeze(processed_sentence, 0)
        processed_sentence = processed_sentence.cuda()
        if self.use_gst:
            mel_outputs, mel_outputs_postnet, gates, alignments = self.model.inference(processed_sentence,
                                                                                       reference_mel=self.ref_mel)
        else:
            mel_outputs, mel_outputs_postnet, gates, alignments = self.model.inference(processed_sentence)
        mel = mel_outputs_postnet[0]
        mel = torch.autograd.Variable(mel.cuda())
        mel = torch.unsqueeze(mel, 0)
        mel = mel.half() if self.is_fp16 else mel
        audio = self.waveglow.infer(mel, sigma=self.sigma)
        audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        return audio

    def synthesize_ssml(self, ssml_text):
        try:
            ssml_elements = parse_ssml_text(ssml_text)
        except SSMLException as e:
            raise e
        except Exception as e:
            raise RuntimeError("Failed to parse ssml")
        audio_segments = []
        for element in ssml_elements:
            ssml_element = ssml_factory(element)
            content, is_text = ssml_element.get_content()
            if is_text:
                content = self.synthesize(content)
            audio = ssml_element.postprocess_content(content)
            audio_segments.append(audio)
        merged_audio = np.concatenate(audio_segments, axis=0).astype(np.int16)
        return merged_audio
