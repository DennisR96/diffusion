from utils.utils import load_yaml_config, dict2namespace


config = load_yaml_config("config/MNIST.yaml")
test = dict2namespace(config)
print(test.model.batch_size)