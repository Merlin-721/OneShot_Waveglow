#!/usr/bin/env python3
"""Create a recording with arbitrary duration.
The soundfile module (https://PySoundFile.readthedocs.io/) has to be installed!
"""
import queue
import sys

import numpy as np
import sounddevice as sd
import soundfile as sf
import numpy
assert numpy  # avoid "imported but unused" message (W0611)


q = queue.Queue()

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def record_audio(samplerate=22050, channels=1, device=None, subtype=None):
	audio = []
	try:
		with sd.InputStream(samplerate=samplerate, device=device, dtype=np.int16,
							channels=channels, callback=callback):
			print('=' * 80)
			print('press Ctrl+C to stop the recording')
			print('=' * 80)
			while True:
				audio.extend(q.get())
				# file.write(q.get())
	except KeyboardInterrupt:
		print('\nRecording finished')
		return audio
	except Exception as e:
		print('Error: ' + repr(e))