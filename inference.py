import os
import torch

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from model.conformer.model_casual import ConformerVad
from model.conformer.stft import StftHandler

from dataset import VadDataset, DataLoader
import time


def select_and_remove_pref(pref: str, d: OrderedDict) -> OrderedDict:
    pref = pref + '.'
    keys = [k for k in d.keys() if k.startwith(pref)]
    result = OrderedDict()
    for key in keys:
        key_trunc = key[len(pref):]
        result[key_trunc] = d[key]
    return result

def mono_plot(sig, vad, fs=16000):
    plt.subplots(1, 1, figsize=(20,4))
    plt.plot([i / fs for i in range(len(sig))], sig, label='Signal')
    plt.plot([i / fs for i in range(len(vad))], max(sig) * vad, label='VAD')
    plt.legend(loc='best')

    plt.savefig('home/tmp/vad_inf_test.png')
    plt.show()


def processing(signal, model, t, device):
    signal = torch.from_numpy(signal)
    signal = signal.to(device)
    signal = signal.unsqueeze(0)

    first_step = 480
    frame = int(t * 1600 / 1000)

    current_frame = signal[:, :first_step]
    rest_signal = signal[:, first_step:]
    array_out = torch.zeros(480).to(device)
    print('processing rest signal')
    t_out = []
    while rest_signal.shape[-1] > 0:
        frame_tmp = rest_signal[:, :frame]
        current_frame = torch.cat((current_frame, frame_tmp), 1)
        start_time = time.time()
        current_pred = model(current_frame)
        print("--- %s seconds ---" % (time.time() - start_time), current_frame.shape)
        t_out.append(time.time() - start_time)
        array_out = torch.cat((array_out, current_pred[0][-frame:]), 0)
        rest_signal = rest_signal[:, frame:]
    print(np.mean(t_out))
    return array_out

def main():
    t = 10

    path_state = '/home/TrainResults/'
    snapshot = torch.load(
        os.path.join(path_state, 'last_snapshot.tar'),
        map_location='cpu'
    )

    device = 'cuda:7'

    conf_kwargs = dict(
        dim=256,
        dim_head=64,
        heads=4,
        ff_mult=2,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.1,
        ff_dropout=0.1,
        conv_dropout=0.1,
        look_ahead=6,
    )

    model = ConformerVad(
        stft=StftHandler(),
        num_layers=12,
        inp_dim=257,
        out_dim=257,
        conformer_kwards=conf_kwargs,)

    state_dict = snapshot['model']
    state_dict = select_and_remove_pref('module', state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)

    root_valdata = '/home/LibriSpeechWav/dev-clean'
    root_noise = '/home/noise_upgrate'

    val_dataset = VadDataset(root_valdata, root_noise)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    signal, gt_mask = next(iter(val_loader))

    test_sig = signal[0].detatch().cpu().numpy()
    pred_mask = processing(test_sig, model, t, device)
    pred_mask = pred_mask.detach().cpu().numpy()
    pred_mask = pred_mask[:test_sig.shape[-1]]
    mono_plot(test_sig, pred_mask)

if __name__ == '__main__':
    main()