import os


class Config(object):
    def __init__(self, output_dir, epochs, batch_size, log_every_step):
        self.ckpt_dir_train = os.path.join(output_dir, 'ckpt/train')
        self.ckpt_dir_dev = os.path.join(output_dir, 'ckpt/dev')
        self.log_dir_train = os.path.join(output_dir, 'log/train')
        self.log_dir_dev = os.path.join(output_dir, 'log/dev')

        self.epochs = epochs
        self.batch_size = batch_size
        self.log_every_step = log_every_step
