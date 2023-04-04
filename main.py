import os
import sys
import argparse
import numpy as np
import numpy as np
from collections import defaultdict


class HMM:
    def __init__(self, states, observations, start_prob, trans_prob, emit_prob):
        self.states = states
        self.observations = observations
        self.start_prob = start_prob
        self.trans_prob = trans_prob
        self.emit_prob = emit_prob

    def viterbi(self, obs):
        # Initialize the trellis
        T = len(obs)
        N = len(self.states)
        trellis = np.zeros((T, N))
        backpointer = np.zeros((T, N), dtype=np.int)

        # Set the initial trellis values
        trellis[0, :] = self.start_prob * self.emit_prob[:, self.observations.index(obs[0])]

        # Fill in the rest of the trellis
        for t in range(1, T):
            for j in range(N):
                prob = trellis[t-1, :] * self.trans_prob[:, j] * self.emit_prob[j, self.observations.index(obs[t])]
                trellis[t, j] = np.max(prob)
                backpointer[t, j] = np.argmax(prob)

        # Find the best sequence of states
        best_sequence = [np.argmax(trellis[-1, :])]
        for t in range(T-1, 0, -1):
            best_sequence.append(backpointer[t, best_sequence[-1]])
        best_sequence.reverse()

        return [self.states[i] for i in best_sequence]

"""# Example usage
states = ['N', 'V', 'D']
observations = ['the', 'dog', 'ran']
start_prob = np.array([0.3, 0.3, 0.4])
trans_prob = np.array([[0.3, 0.3, 0.4],
                       [0.2, 0.5, 0.3],
                       [0.1, 0.1, 0.8]])
emit_prob = np.array([[0.6, 0.1, 0.1],
                      [0.1, 0.2, 0.1],
                      [0.1, 0.2, 0.2]])

hmm = HMM(states, observations, start_prob, trans_prob, emit_prob)
obs = ['the', 'dog', 'ran']
tagged_sequence = hmm.viterbi(obs)
print(tagged_sequence)"""


def read_from_file(filename):
    f = open(filename)
    lines = f.readlines()
    puzzle = [[str(x) for x in l.rstrip()] for l in lines]
    f.close()


def train_hmm(training_file):
    # Initialize dictionaries to store counts

    start_counts = {}
    transition_counts = {}
    emission_counts = {}

    tag_counts = {}
    tag_number=0
    tag_list=[]

    word_counts = {}
    word_number=0
    word_list = []


    for file in training_file:
        f = open(file, "r")
        lines = f.readlines()


    # Process each line in the training file
    for line in lines:
        #print(line)

        prev_tag = '<s>'
        """tokens = line.strip().split()
        print(tokens)"""
        """prev_tag = '<s>'
        if prev_tag not in start_counts:
            start_counts[prev_tag] = 0
        start_counts[prev_tag] += 1"""

        word, tag=line.rsplit(' : ')

        word_list.append(word)
        tag_list.append(tag[:-1])
    #TRANSITION
    transition_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)
    for sequence in tag_counts:
        for i in range(len(sequence) - 1):
            transition_counts[sequence[i]][sequence[i + 1]] += 1
            tag_counts[sequence[i]] += 1
    transition_probabilities = defaultdict(lambda: defaultdict(float))
    for tag1, tag2_counts in transition_counts.items():
        for tag2, count in tag2_counts.items():
            transition_probabilities[tag1][tag2] = count / tag_counts[tag1]
    print(transition_probabilities)

    #print(transition_probabilities)
    #print(transition_probabilities)
    #return transition_probabilities

    """if word not in word_counts:
            #not in dict already
            word_counts[word] = word_number
            word_list.append(word)
            word_list.append(word_number)
            word_number += 1



        if tag not in tag_counts:
            tag_counts[tag] = 0
            tag_counts[tag] += 1


        if tag not in transition_counts:
            transition_counts[tag] = {}
        if tag not in transition_counts[tag]:
            transition_counts[prev_tag][tag] = 0
            transition_counts[prev_tag][tag] += 1


        if tag not in emission_counts:
            emission_counts[tag] = {}


        if word not in emission_counts[tag]:
            emission_counts[tag][word] = 0
            emission_counts[tag][word] += 1
            prev_tag = tag

    # Calculate probabilities
    start_prob = {}
    for tag in start_counts:
        start_prob[tag] = start_counts[tag] / sum(start_counts.values())

    trans_prob = {}
    for prev_tag in transition_counts:
        trans_prob[prev_tag] = {}
        for tag in transition_counts[prev_tag]:
            trans_prob[prev_tag][tag] = transition_counts[prev_tag][tag] / sum(transition_counts[prev_tag].values())

    emit_prob = {}
    for tag in emission_counts:
        emit_prob[tag] = {}
        for word in emission_counts[tag]:
            emit_prob[tag][word] = emission_counts[tag][word] / tag_counts[tag]

    # Return the trained HMM
    states = list(tag_counts.keys())
    observations = list(set([word for tag in emission_counts for word in emission_counts[tag]]))
    hmm = HMM(states, observations, start_prob, trans_prob, emit_prob)


    print(trans_prob['<s>'])
    print(len(trans_prob))
    return hmm
"""

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


    train_hmm(args.trainingfiles[0])

