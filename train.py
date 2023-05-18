import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# Alec Ames
# 6843577
# COSC 4P80 - Term Project

# Drum Sample Captioning Using Deep Learning

# Training Script


# prepare audio and create audio dataset
def prepare_audio_dataset(samples, max_length=16000):
	resampler = torchaudio.transforms.Resample(44100, 16000)
	dataset = []
	for index in range(len(samples)):
		audio_path, label = samples[index]
		waveform, _ = torchaudio.load(audio_path)

		# mixdown to mono
		if waveform.shape[0] > 1:
			waveform = torch.mean(waveform, dim=0, keepdim=True)

		# resample to 16khz
		waveform = resampler(waveform)

		# pad/truncate to 1s
		if waveform.shape[-1] < max_length:
			padding = max_length - waveform.shape[-1]
			waveform = torch.cat([waveform, torch.zeros(waveform.shape[0], padding)], dim=-1)
		elif waveform.shape[-1] > max_length:
			waveform = waveform[:, :max_length]

		dataset.append((waveform, label))
	return dataset


# define CNN model
class CNN(nn.Module):
	def __init__(self, num_classes):
		super(CNN, self).__init__()
		self.model = nn.Sequential(
			nn.Conv1d(1, 32, kernel_size=80, stride=4),
			nn.BatchNorm1d(32),
			nn.ReLU(),
			nn.MaxPool1d(4),
			nn.Conv1d(32, 64, kernel_size=3),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.MaxPool1d(4),
		)
		self.fc1_size = 16000 - 128
		self.fc1 = nn.Linear(self.fc1_size, num_classes)

	def forward(self, x):
		x = self.model(x)
		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		return x


# define labels
def load_dataset(root_dir):
	labels = {
		"kick": 0,
		"snare": 1,
		"tom": 2,
		"cymbal": 3,
		"hat": 4,
	}
	samples = []

	for folder, label in labels.items():
		folder_path = os.path.join(root_dir, "data", folder)
		for file_name in os.listdir(folder_path):
			file_extension = os.path.splitext(file_name)[-1].lower()
			if file_extension == ".wav":
				file_path = os.path.join(folder_path, file_name)
				samples.append((file_path, label))
	return samples


def generate_labels(batch_size, num_classes):
	return torch.randint(0, num_classes, (batch_size,))


root_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# hyperparams
num_classes = 5
num_epochs = 250
batch_size = 64
learning_rate = 0.001

# load data
samples = load_dataset(root_dir)
train_samples, test_samples = train_test_split(samples, test_size=0.2)

train_dataset = prepare_audio_dataset(train_samples)
test_dataset = prepare_audio_dataset(test_samples)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# define model, loss, and optimizer
model = CNN(num_classes).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training loop
epoch_loop = tqdm(range(num_epochs), leave=True, position=0, unit="epoch")
for epoch in epoch_loop:
	running_loss = 0.0
	num_batches = 0

	for i, (audio, labels) in enumerate(train_loader):
		audio, labels = audio.to(device), labels.to(device)

		# fwd pass
		outputs = model(audio)
		loss = loss_function(outputs, labels)

		# backwd and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		num_batches += 1

print(f"Training samples #: {len(train_samples)}")
print(f"Testing samples #: {len(test_samples)}")

# test model
model.eval()
with torch.no_grad():
	correct = 0
	total = 0
	for audio, labels in test_loader:
		audio, labels = audio.to(device), labels.to(device)
		outputs = model(audio)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	print(f"Test Accuracy: {100 * correct / total}%")

torchscript = torch.jit.script(model)
torch.jit.save(torchscript, "model_v0.13.pt")
