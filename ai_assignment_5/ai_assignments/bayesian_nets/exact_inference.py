import hedgehog as hh
import pandas as pd


def build_network():
    bn = ...
    # TODO: build network
    return bn


def initialize_probabilities(bn):
    # TODO: initialize probabilities
    return bn


if __name__ == '__main__':
    bn = build_network()
    # optional: visualize network to check whether the structure is correct

    bn = initialize_probabilities(bn)
    bn.prepare()  # implemented in hedgehog, don't remove

    # TODO: compute intermediate probabilities, alpha and exact probabilities
    # TODO: print results
    # TODO: enter required numbers in Moodle
