#!/usr/bin/env python3
"""Create a recording with arbitrary duration.
The soundfile module (https://PySoundFile.readthedocs.io/) has to be installed!
"""
import queue
import sys

import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)


q = queue.Queue()

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def record_audio(filename, samplerate=22050, channels=1, device=None, subtype=None):
	
	try:
		# Make sure the file is opened before recording anything:
		with sf.SoundFile(filename, mode='x', samplerate=samplerate,
									channels=channels, subtype=subtype) as file:
			with sd.InputStream(samplerate=samplerate, device=device,
								channels=channels, callback=callback):
				print('=' * 80)
				print('press Ctrl+C to stop the recording')
				print('=' * 80)
				while True:
					file.write(q.get())
	except KeyboardInterrupt:
		print('\nRecording finished: ' + repr(filename))
		# parser.exit(0)
	except Exception as e:
		print('Error: ' + repr(e))
		# parser.exit(type(e).__name__ + ': ' + str(e))