from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, tb_path):
        self.writer = SummaryWriter(tb_path)

    def log(self, step, value, name, mode):
        self.writer.add_scalar('{}/{}'.format(mode, name), value, global_step=step)

    def close(self):
        self.writer.close()