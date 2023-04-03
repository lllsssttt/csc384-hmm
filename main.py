import os
import sys
import argparse
import numpy as np


class HMM:
    def __init__(self, num_states, num_obs, start_prob, trans_prob, emit_prob):
        self.num_states = num_states
        self.num_obs = num_obs
        self.start_prob = start_prob
        self.trans_prob = trans_prob
        self.emit_prob = emit_prob

    def forward(self, obs):
        alpha = np.zeros((len(obs), self.num_states))
        alpha[0] = self.start_prob * self.emit_prob[:, obs[0]]

        for t in range(1, len(obs)):
            alpha[t] = np.dot(alpha[t - 1], self.trans_prob) * self.emit_prob[:, obs[t]]

        return alpha

    def backward(self, obs):
        beta = np.zeros((len(obs), self.num_states))
        beta[-1] = 1

        for t in reversed(range(len(obs) - 1)):
            beta[t] = np.dot(self.trans_prob, self.emit_prob[:, obs[t + 1]] * beta[t + 1])

        return beta

    def predict(self, obs):
        alpha = self.forward(obs)
        return alpha[-1].argmax()

    def decode(self, obs):
        alpha = self.forward(obs)
        beta = self.backward(obs)
        prob = alpha[-1].sum()
        state = np.zeros(len(obs), dtype=np.int)
        for t in range(len(obs)):
            state[t] = np.argmax(alpha[t] * beta[t] / prob)
        return state

    def train(self, obs, max_iter=100):
        for n in range(max_iter):
            alpha = self.forward(obs)
            beta = self.backward(obs)
            xi = np.zeros((len(obs) - 1, self.num_states, self.num_states))
            for t in range(len(obs) - 1):
                xi[t] = alpha[t][:, np.newaxis] * self.trans_prob * self.emit_prob[:, obs[t + 1]] * beta[t + 1][
                                                                                                    np.newaxis, :]

            gamma = alpha * beta / alpha[-1]

            self.start_prob = gamma[0]
            self.trans_prob = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, np.newaxis]
            self.emit_prob = np.zeros((self.num_states, self.num_obs))
            for k in range(self.num_obs):
                mask = obs == k
                self.emit_prob[:, k] = gamma[mask].sum(axis=0) / gamma.sum(axis=0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    training_list = args.trainingfiles[0]
    print("training files are {}".format(training_list))

    print("test file is {}".format(args.testfile))

    print("output file is {}".format(args.outputfile))


    print("Starting the tagging process.")

