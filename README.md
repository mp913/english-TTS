#### Usage
1. Download all submodules and install requirements for them.
2. Specify paths for tacotron2 and vocoder models, tacotron2 hparams file, output directory and audio file with GST emotion sample in *yml config file.
3. Specify GST and explicit language usage in *yml config file, if you wish to.
4. Run tts.py script with your python3 using these flags:
    * -p/--config_path - path to the configuration file. By default it is ./tts_config.yml.
    * -t/--text_to_speech - text for synthesis.
    * -n/--file_name - output file name for "-t" flag text. By default it is synthesis.wav.
    * -c/--csv_to_speech - path to .csv-file with texts for synthesis in <output_file_name>|<txt_for_synthesis> format.
    * -s/--use_ssml - flag for SSML usage.

#### Acceptable SSML flags
* <speak> Put your text here </speak> - standard tag for text generation.
* <sub alias=”value”> Text for substitution </sub> - tag for substitution internal text to alias value. 
* <p> Put your text here </p> - tag for paragraph markdown.
* <s> Put your text here </s> - tag for sentences markdowm.
* <break time=”value” strength=””/> - tag for extra breakes in text. Can be used with time (100ms, 1s) attribute, or with one of "x-weak", weak", "medium", "strong", "x-strong" values for strength attribute.
* <prosody pitch=”” volume=”” rate=””> Put your text here </prosody> - tag for pitch volume and speed customization. 
    * pitch can be customized with number (150%) or with one of "x-slow", "slow", "medium", "fast", "x-fast", "default" values.
    * volume can be customized with dB value like “+6dB” or “-12dB” or with one of "silent" (zero volume), "x-soft" (strong volume decrease), "soft"  (volume decrease), "medium" (no change), "loud" -  (volume increase), "x-loud" -  (strong volume increase). 
    * rate can be customized with number (150%) or with one of"x-slow", "slow", "medium", "fast", "x-fast", "default" values.
* <say-as interpret-as="[characters/expletive]"> Put your text here </say-as> - multifunctional tag for additional options of text pronunciation. For now can be used for letter-by-letter (characters value) reading or for the censorship (expletive value).
