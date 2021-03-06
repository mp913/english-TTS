import abc
import pydub
from pysndfx import AudioEffectsChain
import numpy as np
from tps.modules.ssml.elements import Text, Pause, SayAs
from typing import Union


class SSMLException(Exception):
    def __init__(self, text='Unexpected SSML error'):
        super().__init__(text)


class SSMLElement(abc.ABC):
    sample_rate = 22050
    max_wav_value = 32768.0

    def __init__(self, pitch=1.0, rate=1.0, volume=0.0):
        self.pitch = pitch
        self.rate = rate
        self.volume = volume

    @abc.abstractmethod
    def get_content(self):
        pass

    def postprocess_content(self, audio):
        audio = self.change_pitch(audio)
        audio = audio / SSMLElement.max_wav_value
        audio = self.change_rate(audio)
        audio = self.change_volume(audio)
        audio = audio * SSMLElement.max_wav_value
        return audio

    def change_pitch(self, audio):
        audio = audio.astype('int16')
        sound_pydub = pydub.AudioSegment(
            audio.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=audio.dtype.itemsize,
            channels=1
        )
        octaves = self.pitch - 1
        new_sampling_rate = int(sound_pydub.frame_rate * (2.0 ** octaves))
        pitch_sound = sound_pydub._spawn(sound_pydub.raw_data, overrides={'frame_rate': new_sampling_rate})
        pitch_sound = pitch_sound.set_frame_rate(self.sample_rate)
        audio = np.array(pitch_sound.get_array_of_samples()).astype('float')

        return audio

    def change_rate(self, audio):
        fx = (AudioEffectsChain().speed(self.rate))
        audio = fx(audio, sample_in=self.sample_rate)
        return audio

    def change_volume(self, audio):
        audio = audio * (10 ** (self.volume / 20))
        audio = np.clip(audio, -1.0, 1.0)
        return audio


class SSMLBreak(SSMLElement):
    def __init__(self, duration):
        super().__init__()
        self.duration = duration
        if self.duration == 0:
            raise SSMLException(f'Break duration can not be zero')
        frame_count = int(self.duration * SSMLElement.sample_rate)
        self.audio = np.zeros(frame_count, dtype=np.float32)

    def get_content(self):
        return self.audio, False

    def postprocess_content(self, audio):
        return audio


class SSMLSayAs(SSMLElement):
    @staticmethod
    def interpret_as_characters(text: str):
        text = text.upper()
        return ' '.join(list(text)), True

    @staticmethod
    def interpret_as_expletive(text: str):
        duration = SSMLSayAs.symbol_duration * len(text)
        frames = max(int(duration * SSMLElement.sample_rate), 1)
        noise = np.zeros(frames, dtype=np.float32)
        limit = 20000  # amplitude of bleep signal
        freq = 1000  # frequency of bleep signal
        fpp = int(SSMLElement.sample_rate / freq)
        for i in range(len(noise)):
            noise[i] = limit * (1 - 2 * abs(1 - 2 * (float(i % fpp) / fpp)))
        return noise, False

    symbol_duration = 0.1
    interpret_as_mapper = {
        'characters': interpret_as_characters.__func__,
        'expletive': interpret_as_expletive.__func__,
    }

    def __init__(self, interpret_as, content, pitch=1.0, rate=1.0, volume=0.0):
        super().__init__(pitch=pitch, rate=rate, volume=volume)
        self.interpreter_function = SSMLSayAs.interpret_as_mapper.get(interpret_as, None)
        if self.interpreter_function is None:
            raise SSMLException(
                f'Invalid "interpret-as" value: {interpret_as}. Supported values: ["characters", "expletive"]')
        self.interpreted_content, self.is_text = self.interpreter_function(content)

    def get_content(self):
        return self.interpreted_content, self.is_text


class SSMLText(SSMLElement):
    def __init__(self, text, pitch=1.0, rate=1.0, volume=0.0):
        super().__init__(pitch=pitch, rate=rate, volume=volume)
        self.text = text

    def get_content(self):
        return self.text, True


def ssml_factory(element: Union[Text, Pause, SayAs]):
    if isinstance(element, Text):
        return SSMLText(element.value, pitch=element.pitch, rate=element.rate, volume=element.volume)
    if isinstance(element, Pause):
        return SSMLBreak(element.seconds)
    if isinstance(element, SayAs):
        return SSMLSayAs(interpret_as=element.interpret_as, content=element.text, pitch=element.pitch,
                         rate=element.rate, volume=element.volume)

    raise SSMLException("Invalid argument's type")
