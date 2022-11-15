import glob
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

from utils.norm import Transform
from utils.mask_generator import Masking

def random_selection(signal, l):
    len_sig = len(signal)
    start = random.randint(0, len_sig - l)
    end = start + l
    return signal[start:end]

def random_silence(signal, l_silence):
    len_sig = len(signal)
    start = random.randint(0, len_sig - l_silence)
    end = start + l_silence
    signal[start:end] = np.zeros(l_silence, dtype='float32')
    return signal

class VadDataset(Dataset):
    def __init__(self, root_utt, root_noise, train=True):
        self.audio_files_utt = glob.glob(f'{root_utt}/**/*.wav', recursive=True)
        self.audio_files_noises = glob.glob(f'{root_noise}/**/*.wav', recursive=True)
        self.audio_files_noises.sort()
        self.transform = Transform
        self.mask_gen = Masking()
        np.random.RandomState(42).shuffle(self.audio_files_noises)
        train_len = int(0.8 * len(self.audio_files_noises))
        if train:
            self.audio_files_noises = self.audio_files_noises[:train_len]
        else:
            self.audio_files_noises = self.audio_files_noises[train_len:]

        print(f'Number of noises {len(self.audio_files_noises)}')

    def __len__(self):
        return len(self.audio_files_utt)

    def extractor(self, signal_path, noise_path, t_cut=3, t_silence=1):
        signal, sr = sf.read(signal_path, dtype='float32')
        if len(signal) >= t_cut * sr:
            signal = random_selection(signal, t_cut*sr)
            signal = random_silence(signal, t_silence*sr)
        else:
            signal = np.zeros(t_cut * sr, dtype='float32')
        with sf.SoundFile(noise_path) as f:
            frames = f.frames
        if frames >= t_cut * sr + 1:
            noise, sr = sf.read(noise_path, frames=t_cut*sr, start=-frames, dtype='float' )
        else:
            noise, sr = sf.read(noise_path, dtype='float32')
            array_pad = np.zeros(len(signal) - len(noise))
            noise = np.append(noise, array_pad)
        mixture = self.transform(signal, noise)
        mask = self.mask_gen(mixture)
        return mixture, mask

    def __getitem__(self, idx):
        noise_idx = np.random.randint(len(self.audio_files_noises))
        path_audio = self.audio_files_utt[idx]
        path_noise = self.audio_files_noises[noise_idx]
        mixture , mask = self.extractor(path_audio, path_noise)
        return mixture, mask


def main():
    path_utts = ''
    path_noise = ''

    train_dataset = VadDataset(path_utts, path_noise)

    train_loader = DataLoader(train_dataset, batch_zise=1, shuffle=True)
    print(len(train_dataset), len(train_loader))
    mizture, mask = next(iter(train_loader))

if __name__ == '__main__':
    main()
