import argparse
from .dataset import MNIST_CIFAR10

parser = argparse.ArgumentParser(description='')
parser.add_argument('--num_class',
                    help='')
parser.add_argument('--flip_ratio_list',
                    help='')
parser.add_argument('--class_ratio',
                    help='')

args = parser.parse_args()

dataset = MNIST_CIFAR10(args)
