from pathlib import Path


class Config:
    def __init__(self):
        self.weights_path = './data/final.h5'
        self.train_path = './data/garden_stuff/train'
        self.test_path = './data/garden_stuff/test'
        self.seed = 228
        self.epochs = 25
        self.img_width = 64
        self.img_height = 64
        self.batch_size = 32
        self.samples_per_epoch = 8192
        self.validation_steps = 512
        self.filters1 = 128
        self.filters2 = 256
        self.filters3 = 512
        self.filters4 = 256
        self.dense_size1 = 256
        self.dense_size2 = 196
        self.dense_size3 = 128
        self.conv_size1 = 4
        self.conv_size2 = 6
        self.conv_size3 = 8
        self.conv_size4 = 4
        self.pool_size = 2
        self.alternative_pool_size = 2
        self.classes_num = 74
        self.lr = 0.001
        self.lr_decay = 0.01
        self.leaky_rely_alpha = 0.1
        self.prediction_steps = 12000
        self.dropout_rate = 0.4
