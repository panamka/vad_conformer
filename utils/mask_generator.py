import numpy as np
import matplotlib.pyplot as plt

def stride_trick(sig, stride_length, stride_step):
    '''
    apply framing using stride trick
    :param sig: signal array
    :param stride_length (int):  length of the stride
    :param stride_step(int):  stride step
    :return: frames array
    '''

    nrows = ((sig.size - stride_length) // stride_step) + 1
    n = sig.strides[0]
    return np.lib.stride_tricks.as_strided(sig,
                                           shape=(nrows, stride_length),
                                           strides=(stride_step*n, n))

def framing(sig, fs=16_000, win_len=0.025, win_hop=0.01):
    '''
    transform a signal into series of overlapping frames

    :param sig: mono signal array
    :param fs: the sampling rate
    :param win_len: window length in sec
    :param win_hop: step between windows in sec
    :return:
        array of frames
        frame length
    '''

    if win_len < win_hop: print("Parameter Error: win_len < win_hop")

    frame_length = win_len * fs
    frame_step = win_hop * fs
    signal_length = len(sig)
    frames_overlap = frame_length - frame_step

    rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
    pad_signal = np.append(sig, np.array([0] * int(frame_step - rest_samples) * int(rest_samples != 0.)))

    frames = stride_trick(pad_signal, int(frame_length), int(frame_step))
    return frames, frame_length

def _calculate_norm_energy(frames):
    value = np.abs(np.fft.rfft(a=frames, n=len(frames)))**2
    return np.sum(value, axis=-1) / len(frames)**2

class Masking:
    def __init__(
            self,
            threshold=0.000003,
            win_len=0.02,
            win_hop=0.02,
            e0=1e7,
    ):
        self.threshold = threshold
        self.win_len = win_len
        self.win_hop = win_hop
        self.e0 = e0,
    def __call__(self, sig, fs=16_000):
        frames, frames_len = framing(sig=sig, fs=fs, win_len=self.win_len, win_hop=self.win_hop)
        energy = _calculate_norm_energy(frames)

        energy = np.repeat(energy, frames_len)
        mask = np.array(energy > self.threshold, dtype=sig.dtype)
        return mask


def mono_plot(sig, vad_mask, fs=16000):
    plt.subplot(1, 1, figsize=(20,4))
    plt.plot([i / fs for i in range(len(sig))], sig, label='Signal')
    plt.plot([i / fs for i in range(len(vad_mask))], max(sig)*vad_mask, label='VAD_mask')

    # plt.savefig('')
    plt.show()

def main():
    sig = np.random.rand(3*16000)
    process = Masking()
    mask = process(sig)
    print(mask.shape)

if __name__ == '__main__':
    main()