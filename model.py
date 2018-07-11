from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, Activation, LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import initializers
from keras.models import Sequential
import utils
import datetime
from config import Config
from sklearn import metrics


def get_model(config: Config):

    model = Sequential()

    model.add(
        Convolution2D(config.filters1,
                      config.conv_size1,
                      config.conv_size1,
                      border_mode="valid",
                      input_shape=(config.img_width, config.img_height, 3),
                      kernel_initializer=initializers.glorot_normal(seed=config.seed)))
    model.add(LeakyReLU(alpha=config.leaky_rely_alpha))
    model.add(MaxPooling2D(pool_size=(config.pool_size, config.pool_size)))

    model.add(Convolution2D(config.filters2,
                            config.conv_size2, config.conv_size2, border_mode="valid",
                            kernel_initializer=initializers.glorot_normal(seed=config.seed)))
    model.add(LeakyReLU(alpha=config.leaky_rely_alpha))
    model.add(MaxPooling2D(pool_size=(config.pool_size, config.pool_size)))

    model.add(Flatten())

    model.add(Dense(config.dense_size1,
                    kernel_initializer=initializers.glorot_normal(seed=config.seed)))
    model.add(LeakyReLU(alpha=config.leaky_rely_alpha))
    model.add(Dropout(config.dropout_rate))

    model.add(Dense(config.dense_size2,
                    kernel_initializer=initializers.glorot_normal(seed=config.seed)))
    model.add(LeakyReLU(alpha=config.leaky_rely_alpha))
    model.add(Dropout(config.dropout_rate))

    model.add(Dense(config.dense_size3,
                    kernel_initializer=initializers.glorot_normal(seed=config.seed)))
    model.add(LeakyReLU(alpha=config.leaky_rely_alpha))
    model.add(Dropout(config.dropout_rate))

    model.add(Dense(config.classes_num, activation='softmax',
                    kernel_initializer=initializers.glorot_normal(seed=config.seed)))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.adam(lr=config.lr, decay=config.lr_decay),
                  metrics=['accuracy'])

    return model


def train_model(config: Config, model: Sequential,
                train_data_generator, validation_data_generator):

    history = model.fit_generator(
        train_data_generator,
        samples_per_epoch=config.samples_per_epoch,
        epochs=config.epochs,
        validation_data=validation_data_generator,
        validation_steps=config.validation_steps,
        use_multiprocessing=True)

    utils.create_dir('data', 0o777)

    print('Saving weights')

    model.save_weights('data/lastweights.h5')

    now = datetime.datetime.now()
    weights_path = './data/weights_' + now.strftime("%Y%m%d%H%M") + '.h5'
    model.save_weights(weights_path)

    print('Saved!')

    return history


def predict(model: Sequential, validation_data_generator, config: Config):
    print('Started evaluation')

    scores = model.evaluate_generator(validation_data_generator, steps=validation_data_generator.n/config.batch_size, verbose=1)

    print('Evaluation completed')

    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    print('Started prediction')

    y_pred = model.predict_generator(validation_data_generator, steps=validation_data_generator.n/config.batch_size, verbose=1)
    y_pred = y_pred.argmax(axis=1)
    y_true = validation_data_generator.classes.reshape(
        validation_data_generator.n, 1)
    print('Completed prediction')

    print('Started calculating metrics')

    accuracy = metrics.accuracy_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    print('Metrics calculated!')

    print('accuracy=')
    print(accuracy)
    print('recall=')
    print(recall)
    print('precision=')
    print(precision)
    print('confusion matrix=')
    print(confusion_matrix)


def load_model(path: str, model: Sequential):
    model.load_weights(path)
