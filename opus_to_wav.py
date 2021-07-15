from soundfile import SoundFile
from Waveglow.mel2samp import Mel2Samp
import librosa
import soundfile

# path = "input_wavs/casey_short.opus"
# output = "input_wavs/casey_short.wav"
path = "input_wavs/casey_30s.opus"
output = "input_wavs/casey_30s.wav"
# path = "input_wavs/draze_midi.opus"
# output = "input_wavs/draze_midi.wav"


sr = 22050
audio,_ = librosa.load(path,sr=sr)
soundfile.write(output,audio,sr)
