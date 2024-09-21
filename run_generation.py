import sys
from omegaconf import OmegaConf
from synthEHRella.data.data_generator import DataGenerator


def main():
    config_file = sys.argv[1]
    config = OmegaConf.load(config_file)
    generator = DataGenerator(config.generation.method)
    generator.generate(config.generation.params)

if __name__ == "__main__":
    main()
