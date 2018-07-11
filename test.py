import model
import utils
from config import Config
import numpy
from sklearn import metrics

config = Config()
numpy.random.seed(config.seed)
cnn = model.get_model(config)

model.load_model(config.weights_path, cnn)

#replace test path to test on different data
y = utils.create_validation_generator(config, config.test_path)

model.predict(cnn, y, config)