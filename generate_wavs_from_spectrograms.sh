#!/bin/bash

cd waveglow
python3 inference.py -f ./file_paths.txt -w /home/max/TTS/pycharm-sova/sova-tts-vocoder/checkpoints/waveglow_256channels_universal_v5.pt -o output
cd -
