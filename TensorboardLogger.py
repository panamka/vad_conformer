from torch.utils.tensorboard import SummaryWriter
class TensorboardLogger:
    def __init__(self, tensorboard_path):
        self.wtriter = SummaryWriter(tensorboard_path)

    def log(self, step, value, name, mode):
        self.wtriter.add_scalar('{}/{}'.format(mode, name), value, global_step=step)

    def close(self):
        self.wtriter.close()
