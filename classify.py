import argparse
import os

import librosa
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

# Alec Ames
# 6843577
# COSC 4P80 - Term Project

# Drum Sample Captioning Using Deep Learning

# Classifier Script


# transient detection with librosa
def detect_transients(waveform, rate, sensitivty=0.1):
	waveform = np.concatenate([np.zeros(int(0.1 * rate)), waveform])

	# calculates the transient envelope
	env = librosa.onset.onset_strength(y=waveform, sr=rate)
	times = librosa.frames_to_time(np.arange(len(env)), sr=rate)
	transients = librosa.onset.onset_detect(onset_envelope=env, sr=rate, delta=sensitivty)

	# only detects if at least 100ms since last transient
	min_diff = 0.1
	last_timestamp = -1
	filtered_transients = []
	for i, transient in enumerate(times[transients]):
		transient -= 0.128
		if i == 0 or transient - last_timestamp >= min_diff:
			filtered_transients.append(transient)
			last_timestamp = transient

	return np.array(filtered_transients)


# resamples and mixes down audio
def prepare_audio(waveform, original_rate):

	# mixdown to mono
	if waveform.ndim > 1:
		waveform = np.mean(waveform, axis=1)

	# resamples to 16khz
	resampled = librosa.resample(waveform, orig_sr=original_rate, target_sr=16000)

	# converts to torch tensor
	resampled_waveform = torch.from_numpy(resampled).float()

	return resampled_waveform


# pads the window to the target length for the model (1s)
def pad(window, rate):
	target_length = rate
	length = window.size(-1)
	if length < target_length:
		pad_amount = target_length - length
		window = torch.nn.functional.pad(window, (0, pad_amount))
	return window


# gpu acceleration if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loads latest model
classify = torch.jit.load("model_v0.12.pt").to(device)
classify.eval()

# define labels
labels = {
	0: "kick",
	1: "snare",
	2: "tom",
	3: "cymbal",
	4: "hat",
}


def main():
	parser = argparse.ArgumentParser(description="caption drum samples in an audio file.")
	parser.add_argument("audio_file_path", help="path to the audio file")
	parser.add_argument("-e", "--export_path",
	                    help="path to export the classified samples; if not specified, samples will not be exported \n e.g. -e \"./export/test/\"")
	args = parser.parse_args()

	audio_file_path = args.audio_file_path
	export_path = args.export_path
	delta = 0.1
	resample_rate = 16000

	print(f"Using device: {device}")

	# load and transform the audio file
	original_waveform, original_rate = librosa.load(audio_file_path, sr=None)
	waveform = prepare_audio(original_waveform, original_rate)

	print(f"Loaded {audio_file_path}")
	print(f"Resampling to {resample_rate}Hz")

   	# get the transients
	transients = detect_transients(waveform.numpy(), resample_rate, delta)
	print(f"Detected " + str(len(transients)) + " transients")

	results = []

	# classify each transient
	for i in range(len(transients)):
		start_sample = int(transients[i] * resample_rate)

		end_sample = start_sample + resample_rate

		if end_sample > waveform.size(-1):
			window = F.pad(waveform[start_sample:], (0, end_sample - waveform.size(-1))).to(device)
		else:
			window = waveform[start_sample:end_sample].to(device)

		# classify the window
		with torch.no_grad():
			output = classify(window.unsqueeze(0).unsqueeze(0))
			probabilities = F.softmax(output, dim=1).squeeze()
			predicted = torch.argmax(probabilities)

			results.append([transients[i], labels[predicted.item()], probabilities[predicted].item()])

		print(f"{transients[i]:.3f}″: {labels[predicted.item()]} ({probabilities[predicted].item() * 100:.2f}%)")

	# export the samples as individual labeled wav files
	if export_path:
		os.makedirs(export_path, exist_ok=True)
		for i, result in enumerate(results):
			start_sample = int(result[0] * original_rate)
			if i < len(results) - 1:
				end_sample = int(results[i + 1][0] * original_rate)
			else:
				end_sample = len(original_waveform)
			slice = original_waveform[start_sample:end_sample]
			export_file_path = os.path.join(export_path, f"{i+1:02d}-{result[1]}.wav")
			# export the slice
			sf.write(export_file_path, slice, original_rate)
		print(f"\nExported {len(results)} samples to {export_path}")

	# plot the results
	plt.figure(figsize=(15, 5))
	plt.title(f"Classification of samples in {audio_file_path}")

	time = np.arange(0, waveform.size(-1)) / resample_rate
	plt.plot(time, waveform.numpy(), linewidth=0.5)
	plt.xlabel("Time (s)")
	plt.ylabel("Amplitude")
	plt.xticks(np.arange(0, time[-1], 0.5))

	for result in results:
		if result[1] == "kick":
			plt.axvline(x=result[0], color='red')
		elif result[1] == "snare":
			plt.axvline(x=result[0], color='magenta')
		elif result[1] == "tom":
			plt.axvline(x=result[0], color='purple')
		elif result[1] == "hat":
			plt.axvline(x=result[0], color='orange')
		elif result[1] == "cymbal":
			plt.axvline(x=result[0], color='olive')

	# create legends for the plot
	kick_legend = mlines.Line2D([], [], color='red', markersize=15, label='Kick')
	snare_legend = mlines.Line2D([], [], color='magenta', markersize=15, label='Snare')
	tom_legend = mlines.Line2D([], [], color='purple', markersize=15, label='Tom')
	hat_legend = mlines.Line2D([], [], color='orange', markersize=15, label='Hat')
	cymbal_legend = mlines.Line2D([], [], color='olive', markersize=15, label='Cymbal')

	plt.legend(handles=[kick_legend, snare_legend, tom_legend, hat_legend, cymbal_legend])
	plt.show()


if __name__ == "__main__":
	main()
