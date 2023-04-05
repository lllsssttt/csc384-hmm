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
    #transition_counts=[[0 for _ in range(91)] for _ in range(91)]
    #initial_counts=[0 for _ in range(91)]
    #emission_counts=[[0 for _ in range(len(word_id))] for _ in range(91)]
    transition_counts=np.zeros(shape=(91,91))
    initial_counts=np.zeros(shape=91)
    emission_counts=np.zeros(shape=(91, len(word_id)))

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

    sentance_ids = [word_id.get(word, None) for word in sentance]
    k=2 #k smoothing to account for unseen words
    Slength=len(sentance)


    prob=[[0 for _ in range(91)] for _ in range(Slength)]
    prev=[[0 for _ in range(91)] for _ in range(Slength)]

    #determine values for time 0
    for i in range(0,91):
        if sentance_ids[0] is not None:
            prob[0][i] = I[i] * E[i, sentance_ids[0]]
        else:
            prob[0][i] = I[i] * (k / (len(word_id) + k * len(tag_id)))
        prev[0][i] = None

    etimeSum=0
    itimeSum=0

    beta=0.1
    row_sums = np.sum(E > 0, axis=1)
    denominators = row_sums + k * len(tag_id)

    #for steps 1 - end of sentance find each state's current state's
    #most likely prior state x
    for t in range(1, Slength):
        for i in range(0, 91):
            max_prob = 0
            max_index = 0
            #x=row_sums[i]
            for j in range(0, 91):
                """
                temp_prob = prob[t - 1][j] * T[j, i]
                if sentance_ids[t] is not None:
                    start_time = time.time()

                    temp_prob *= E[i, sentance_ids[t]]

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    itimeSum += elapsed_time

                    print("If Elapsed time: ", elapsed_time, "TimeSum: ", itimeSum)
                else:
                    start_time = time.time()

                    temp_prob *= (k / (sum(E[i, :] > 0) + k * len(tag_id)))
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    etimeSum += elapsed_time


                    print("Else Elapsed time: ", elapsed_time, "TimeSum: ", etimeSum)
                if temp_prob > max_prob:
                    max_prob = temp_prob
                    max_index = j"""

                temp_prob = prob[t - 1][j] * T[j, i]
                if sentance_ids[t] is not None:
                    temp_prob *= E[i, sentance_ids[t]]
                else:
                    temp_prob *= ((k / (denominators[i])))
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

    if len(SWordIds)!=0:
        Stags += viterbi(T, I, E, SWordIds, tag_id, word_id)
        SWordIds.clear()

    print(len(Stags),len(word_list))

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
    #

    T_normalized,I_normalized,E_normalized,tag_id, word_id=train_hmm(args.trainingfiles[0])
    tagging(T_normalized,I_normalized,E_normalized,tag_id, word_id,args.testfile, args.outputfile)

    with open("sol.txt", "r") as output_file, \
                open("solution2.txt", "r") as solution_file, \
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