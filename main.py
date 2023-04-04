import os
import sys
import argparse
import numpy as np

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

def train_hmm(training_file):
    # Initialize

    initial_counts = []
    transition_counts = []
    emission_counts = []

    tag_counts = {}
    tag_number=0
    tag_list=[]
    tag_id = {}
    tagIdCount=0

    word_counts = {}
    word_number=0
    word_list = []
    word_id={}
    wordIdCount=0

    for file in training_file: #read file
        f = open(file, "r")
        lines = f.readlines()

    # Process each line in the training file
    for line in lines:
        #splitting line into word and tag
        word, tag=line.rsplit(' : ')

        #getting rid of the \n
        tag=tag[:-1]

        #adding the word to the word sequence and indexing
        word_list.append(word)
        if word not in word_id:
            word_counts[word]=1
            word_id[word] = wordIdCount
            wordIdCount += 1
        else:
            word_counts[word]+=1

        #adding the tag to the tag sequence
        tag_list.append(tag)
        #adding to tag count and indexing
        if tag not in tag_counts:
            tag_counts[tag]=1
            tag_id[tag]=tagIdCount
            tagIdCount+=1
        else:
            tag_counts[tag]+=1

    #sentance arrays
    SWordIds=[]
    STagIds=[]

    #create matricies
    transition_counts=np.zeros(shape=(91,91))
    initial_counts=np.zeros(shape=(91))
    emission_counts=np.zeros(shape= (91,(len(word_id))))

    #trans probs
    for i in range(0, len(word_list)):
        SWordIds.append(word_list[i])
        STagIds.append(tag_list[i])

        if word_list[i] in [".", "?", "!"]:
            Slength=len(STagIds)  #sentance length

            #calculate all three probailities and save to respective matricies

            initial_counts[tag_id[STagIds[0]]]+=1

            for j in range(Slength-1):
                transition_counts[tag_id[STagIds[j]]][tag_id[STagIds[j+1]]]+=1

            for j in range(Slength):
                emission_counts[tag_id[STagIds[j]]][word_id[SWordIds[j]]]+=1

            #clear the sentance arrays
            SWordIds.clear()
            STagIds.clear()

    #normalize arrays
    T=np.array(transition_counts)
    row_sums = np.sum(T, axis=1)
    T_normalized = T / row_sums[:, np.newaxis]

    I = np.array(initial_counts)
    norm = np.linalg.norm(I, ord=1)
    I_normalized = I / norm

    E = np.array(emission_counts)
    row_sums = np.sum(E, axis=1)
    E_normalized = E / row_sums[:, np.newaxis]

    return T_normalized,I_normalized,E_normalized


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

