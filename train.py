from config import Config
import numpy
import utils
import model
import visualization

config = Config()
numpy.random.seed(config.seed)
cnn = model.get_model(config)
X, y_train_valid = utils.create_image_generators(config)
history = model.train_model(config, cnn, X, y_train_valid)


y = utils.create_validation_generator(config, config.test_path)

model.predict(cnn, y, config)

print('Started visualisation')

visualization.show_accuracy_plot(history)
visualization.show_loss_plot(history)




