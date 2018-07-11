from config import Config
from keras.preprocessing.image import ImageDataGenerator
import os


def create_dir(path, mode):
    if not path or os.path.exists(path):
        return []
    (head, tail) = os.path.split(path)
    res = create_dir(head, mode)
    os.mkdir(path)
    os.chmod(path, mode)
    res += [path]
    return res


def create_image_generators(config: Config):
    return create_generator_internal(config.train_path, config.seed, config.img_width, config.img_height, config.batch_size),\
           create_generator_internal(config.test_path, config.seed, config.img_width, config.img_height, config.batch_size)


def create_generator_internal(path: str, seed, width, height, batch_size):
    data = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

    return data.flow_from_directory(
        path,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=seed)


def create_validation_generator(config: Config, path: str):
    data = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True)

    return data.flow_from_directory(
        path,
        target_size=(config.img_width, config.img_height),
        batch_size=config.batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=config.seed)
