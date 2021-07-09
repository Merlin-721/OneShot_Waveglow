ls /LJSpeech-1.1/wavs/*.wav | tail -n+10 > train_files.txt
ls /LJSpeech-1.1/wavs/*.wav | head -n10 > test_files.txt
mkdir checkpoints