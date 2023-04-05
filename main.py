import os
import sys
import argparse
import numpy as np
import time


np.set_printoptions(threshold=sys.maxsize)

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

    lines=[]

    for file in training_file: #read files
        f = open(file, "r")
        lines += f.readlines()

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



    return T_normalized,I_normalized,E_normalized,tag_id, word_id

def viterbi(T,I,E,sentance, tag_id, word_id):
    prob=[]
    prev=[]

    sentance_ids = [word_id.get(word, -4) for word in sentance]

    k=1 #k smoothing to account for unseen words
    Slength=len(sentance)

    prob=[[0 for _ in range(91)] for _ in range(Slength)]
    prev=[[0 for _ in range(91)] for _ in range(Slength)]

    #determine values for time 0
    for i in range(0,91):
        prob[0][i] = I[i] * E[i, sentance_ids[0]]
        prev[0][i] = None

    #for steps 1 - end of sentance find each state's current state's
    #most likely prior state x
    for t in range(1, Slength):
        for i in range(0, 91):
            max_prob = 0
            max_index = 0
            for j in range(0, 91):
                temp_prob = prob[t - 1][j] * T[j, i] * E[i, sentance_ids[t]]
                if temp_prob > max_prob:
                    max_prob = temp_prob
                    max_index = j
            prob[t][i] = max_prob
            prev[t][i] = max_index

    optimal_tags = []
    max_prob = 0
    max_index = 0
    for i in range(0, 91):
        if prob[Slength - 1][i] > max_prob:
            max_prob = prob[Slength - 1][i]
            max_index = i
    for i in range(Slength - 1, -1, -1):
        # Append the most likely tag for the current word to the list.
        max_indexN = [key for key, value in tag_id.items() if value == max_index]
        optimal_tags.append(tag_id[max_indexN[0]])
        # Update the index of the most likely tag for the previous word.
        max_index = prev[i][int(max_index)]
        assert max_index != -1

    Stags=[]
    for i in optimal_tags:
        max_indexN = [key for key, value in tag_id.items() if value == i]
        Stags.append(max_indexN[0])
    Stags.reverse()

    return Stags

def Fakeviterbi(T, I, E, sentence, tag_id, word_id):
    # Look up the indices for the sentence words.
    sentence_ids = [word_id.get(word, -4) for word in sentence]

    # Initialize arrays for the forward probabilities and backpointers.
    num_tags = len(tag_id)
    num_words = len(sentence)
    probs = [[0 for _ in range(len(tag_id))] for _ in range(sentence)]
    pointers = np.zeros((num_words, num_tags), dtype=np.int)

    # Set the initial probabilities.
    probs[0, :] = I * E[:, sentence_ids[0]]

    # Iterate over the remaining words.
    for i in range(1, num_words):
        # Compute the element-wise product of the previous probabilities,
        # the transition matrix, and the emission probabilities.
        products = probs[i-1, :, np.newaxis] * T * E[:, sentence_ids[i]]

        # Find the maximum product for each tag.
        max_products = np.max(products, axis=1)

        # Update the probabilities and pointers.
        probs[i, :] = max_products
        pointers[i, :] = np.argmax(products, axis=1)

    # Find the tag with the highest probability at the end of the sentence.
    end_tag = np.argmax(probs[-1, :])

    # Backtrack through the pointers to find the optimal tag sequence.
    tags = np.zeros(num_words, dtype=np.int)
    tags[-1] = end_tag
    for i in range(num_words-2, -1, -1):
        tags[i] = pointers[i+1, tags[i+1]]

    # Convert the tag indices to tag names and return.
    tag_names = [list(tag_id.keys())[list(tag_id.values()).index(tag)] for tag in tags]
    return tag_names
def tagging(T, I, E, tag_id, word_id, testfile, outputfile):

    word_list = []
    newWords={}
    f = open(testfile, "r")
    lines = f.readlines()
    SWordIds=[]
    STagIds=[]
    Stags=[]

    for line in lines:
        line=line[:-1]
        word_list.append(line) #getting rid of \n

    timeSum=0
    for i in range(0, len(word_list)):
        SWordIds.append(word_list[i])
        #STagIds.append(tag_list[i])
        if word_list[i] in [".", "?", "!"]:
            start_time = time.time()

            Stags+=viterbi(T,I,E,SWordIds, tag_id, word_id)
            end_time = time.time()
            elapsed_time = end_time - start_time
            timeSum+=elapsed_time
            print("Elapsed time: ", elapsed_time, "TimeSum: ", timeSum)

            SWordIds.clear()


    x = open(outputfile, "w")
    for Idx in range(0, len(word_list)):
        x.write("{} : {}\n".format(word_list[Idx], Stags[Idx]))

    return 0

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
    #python3 main.py --trainingfiles training1.txt training2.txt --testfile test1.txt --outputfile sol.txt

    T_normalized,I_normalized,E_normalized,tag_id, word_id=train_hmm(args.trainingfiles[0])
    tagging(T_normalized,I_normalized,E_normalized,tag_id, word_id,args.testfile, args.outputfile)

    with open("sol.txt", "r") as output_file, \
                open("solution1.txt", "r") as solution_file, \
                open("results1.txt", "w") as results_file:
            # Each word is on a separate line in each file.
            output = output_file.readlines()
            solution = solution_file.readlines()
            total_matches = 0

            # generate the report
            for index in range(len(output)):
                if output[index] != solution[index]:
                    results_file.write(f"Line {index + 1}: "
                                       f"expected <{output[index].strip()}> "
                                       f"but got <{solution[index].strip()}>\n")
                else:
                    total_matches = total_matches + 1

            # Add stats at the end of the results file.
            results_file.write(f"Total words seen: {len(output)}.\n")
            results_file.write(f"Total matches: {total_matches}.\n")