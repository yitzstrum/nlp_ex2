import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from mle import Mle
from bigram_hmm import BigramHMM
import plotly.express as px


def main():
    # nltk.download('brown')
    # nltk.download('averaged_perceptron_tagger')
    corpus = nltk.corpus.brown
    corpus = corpus.tagged_sents(categories='news')
    new_corpus = []
    for sentence in corpus:
        new_sentence = [(word, tag.split('-')[0].split('+')[0]) for word, tag
                        in
                        sentence]
        new_corpus.append(new_sentence)

    train_set = new_corpus[:int(len(corpus) * 0.9)]
    test_set = new_corpus[int(len(corpus) * 0.9):]

    mle = Mle(train_set)
    mle.train()
    known_error_rate, unknown_error_rate, total_error_rate = \
        mle.calculate_error_rate(test_set)

    print("------------------------b.ii------------------------")
    print("MLE calculation")
    print(f"unknown error: {unknown_error_rate}")
    print(f"known error: {known_error_rate}")
    print(f"total error rate: {total_error_rate}")
    print("----------------------------------------------------")

    bigram_hmm = BigramHMM(train_set)
    bigram_hmm.train()
    known_error_rate, unknown_error_rate, total_error_rate = \
        bigram_hmm.calculate_error_rate(test_set)

    print("------------------------c.iii------------------------")
    print("regular bigram HMM")
    print(f"unknown error: {unknown_error_rate}")
    print(f"known error: {known_error_rate}")
    print(f"total error rate: {total_error_rate}")
    print("----------------------------------------------------")

    bigram_hmm_add_one = BigramHMM(train_set, 1)
    bigram_hmm_add_one.train()
    known_error_rate, unknown_error_rate, total_error_rate = \
        bigram_hmm_add_one.calculate_error_rate(test_set)

    print("------------------------d.ii------------------------")
    print("Laplace Add-one smoothing")
    print(f"unknown error: {unknown_error_rate}")
    print(f"known error: {known_error_rate}")
    print(f"total error rate: {total_error_rate}")
    print("----------------------------------------------------")

    bigram_hmm_pseudo = BigramHMM(train_set, is_pseudo=True)
    bigram_hmm_pseudo.train()
    known_error_rate, unknown_error_rate, total_error_rate = \
        bigram_hmm_pseudo.calculate_error_rate(test_set)

    print("------------------------e.ii------------------------")
    print("Pseudo words")
    print(f"unknown error: {unknown_error_rate}")
    print(f"known error: {known_error_rate}")
    print(f"total error rate: {total_error_rate}")
    print("----------------------------------------------------")
    #
    # bigram_hmm_pseudo_laplace = BigramHMM(train_set, 1, True, True)
    # bigram_hmm_pseudo_laplace.train()
    # known_error_rate, unknown_error_rate, total_error_rate, confusion_matrix = \
    #     bigram_hmm_pseudo_laplace.calculate_error_rate(test_set)
    #
    # print("------------------------e.iii------------------------")
    # print("Pseudo words")
    # print(f"unknown error: {unknown_error_rate}")
    # print(f"known error: {known_error_rate}")
    # print(f"total error rate: {total_error_rate}")
    # print(confusion_matrix)
    # cfm = px.imshow(confusion_matrix)
    # cfm.show()
    # print("----------------------------------------------------")
    #
    #



if __name__ == '__main__':
    main()
