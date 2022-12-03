from collections import defaultdict
import re

import nltk
from sklearn.metrics import confusion_matrix


class BigramHMM:

    def __init__(self, train_set, delta=0, is_pseudo=False,
                 calc_confusion_matrix=False):
        self.train_set = train_set
        self.word_dict = defaultdict(int)
        self.emission_prob = dict()
        self.transition_prob = dict()
        self.unknown_tag = "NN"
        self.delta = delta
        self.is_pseudo = is_pseudo
        self.calc_confusion_matrix = calc_confusion_matrix
        self.words_to_remove = set()

    def train(self):
        self.count_words()
        self.train_emission()
        self.train_transition()

    def count_words(self):
        for sentence in self.train_set:
            for word, tag in sentence:
                self.word_dict[word] += 1

    def get_pseudo(self, word):
        pseudo_words = [(r'^\d{2}$', 'two_digits_num'),
                        (r'^\d{4}$', 'four_digits_num'),
                        (r'^[A-Z]\d+-\d+$', 'product_code'),
                        (r'^\'\d{2}', 'date'),
                        (r'^\d{1,2}:\d{2}(:\d{2})?$', 'time'),
                        (r'^[A-Z]+$', 'all_caps'),
                        (r'^[a-z]+$', 'all_lower'),
                        (r'^\w+tion$', 'tion_suffix'),
                        (r'^\w+re$', 're_suffix'),
                        (r'^\w+end$', 'end_suffix'),
                        (r'^\w+er$', 'er_suffix'),
                        (r'^\w+ing$', 'ing_suffix'),
                        (r'^\w+ed$', 'ed_suffix'),
                        (r'^\$(\d|,)+(\.\d+)?$', 'price'),
                        (r'^pre\w+$', 'pre_prefix'),
                        (r'^sub\w+$', 'sub_prefix'),
                        (r'^re\w+$', 're_prefix'),
                        (r'^in\w+$', 'in_prefix')]
        for reg, p_word in pseudo_words:
            if re.search(reg, word):
                self.word_dict[p_word] += 1
                self.words_to_remove.add(word)
                return p_word
        return word

    def train_emission(self):
        for sentence in self.train_set:
            for word, tag in sentence:
                if self.is_pseudo and self.word_dict[word] < 5:
                    word = self.get_pseudo(word)
                if tag not in self.emission_prob:
                    self.emission_prob[tag] = {word: 1}
                else:
                    if word not in self.emission_prob[tag]:
                        self.emission_prob[tag][word] = 1
                    else:
                        self.emission_prob[tag][word] += 1

        for tag in self.emission_prob:
            tag_sum = sum(self.emission_prob[tag].values())
            for word, count in self.emission_prob[tag].items():
                self.emission_prob[tag][word] = (count + self.delta) / (
                            tag_sum + (
                                self.delta * len(self.emission_prob[tag])))

        # in pseudo words we delete all the original words that we converted
        for word in self.words_to_remove:
            self.word_dict.pop(word, None)

    def train_transition(self):
        for sentence in self.train_set:
            prev_tag = "START"
            for _, tag in sentence:
                if prev_tag not in self.transition_prob:
                    self.transition_prob[prev_tag] = {tag: 1}
                else:
                    if tag not in self.transition_prob[prev_tag]:
                        self.transition_prob[prev_tag][tag] = 1
                    else:
                        self.transition_prob[prev_tag][tag] += 1
                prev_tag = tag
            if prev_tag not in self.transition_prob:
                self.transition_prob[prev_tag] = {"STOP": 1}
            else:
                if "STOP" not in self.transition_prob[prev_tag]:
                    self.transition_prob[prev_tag]["STOP"] = 1
                else:
                    self.transition_prob[prev_tag]["STOP"] += 1
        for tag in self.transition_prob:
            tag_sum = sum(self.transition_prob[tag].values())
            for next_tag, count in self.transition_prob[tag].items():
                self.transition_prob[tag][next_tag] = count / tag_sum

    def get_tag_sequence(self, bp, max_ind, ordered_tags):
        tag_sequence = [ordered_tags[max_ind]]
        for col in range(len(bp[0]) - 1, 0, -1):
            max_ind = bp[max_ind][col]
            tag_sequence = [ordered_tags[max_ind]] + tag_sequence
        return tag_sequence

    def calculate_single_emission_prob(self, word, tag):
        word = self.get_pseudo(word) if (self.is_pseudo and
                                         (word not in self.word_dict or
                                          self.word_dict[word] < 5)) else word

        if word not in self.word_dict:
            if tag == self.unknown_tag:
                return 1
            else:
                tag_sum = sum(self.emission_prob[tag].values())
                return self.delta / (tag_sum +
                                     (self.delta * len(
                                         self.emission_prob[tag])))
        else:

            if word in self.emission_prob[tag]:
                emission_prob = self.emission_prob[tag][word]
            else:
                emission_prob = 0
        return emission_prob

    def viterbi(self, sentence):
        # if word is not in the train set then the emission is 1
        ordered_tags = [tag for tag in self.emission_prob if tag != "START"]
        bp = [[0 for _ in range(len(sentence))] for _ in
              range(len(ordered_tags))]
        pi_prev_k = [1 for _ in range(len(ordered_tags))]
        pi_k = []
        for word_ind in range(len(sentence)):
            for tag_ind, tag in enumerate(ordered_tags):
                max_prob = 0
                max_ind = 0
                for prev_tag_ind, prev_tag in enumerate(ordered_tags):
                    if word_ind == 0:
                        prev_tag = "START"
                    transition_prob = self.transition_prob[prev_tag].get(tag,
                                                                         0)
                    word = sentence[word_ind][0]
                    emission_prob = self.calculate_single_emission_prob(word,
                                                                        tag)
                    curr_prob = pi_prev_k[
                                    prev_tag_ind] * transition_prob * emission_prob
                    if curr_prob > max_prob:
                        max_prob = curr_prob
                        max_ind = prev_tag_ind
                bp[tag_ind][word_ind] = max_ind
                pi_k.append(max_prob)
            pi_prev_k = pi_k
            pi_k = []

        for ind, prev_tag in enumerate(ordered_tags):
            transition_prob = self.transition_prob[prev_tag].get("STOP", 0)
            pi_k.append(pi_prev_k[ind] * transition_prob)

        max_prob = max(pi_k)
        max_ind = pi_k.index(max_prob)
        return self.get_tag_sequence(bp, max_ind, ordered_tags)

    def calculate_error_rate(self, test_set):
        if self.calc_confusion_matrix:
            y_true = []
            y_predicted = []
        unknown_errors = 0
        known_errors = 0
        total_known = 0
        total_unknown = 0
        for sentence in test_set:
            predict_tags = self.viterbi(sentence)
            if self.calc_confusion_matrix:
                y_true += [tag for word, tag in sentence]
                y_predicted += predict_tags
            for i, (word, tag) in enumerate(sentence):
                if word not in self.word_dict:
                    total_unknown += 1
                    if tag != predict_tags[i]:
                        unknown_errors += 1
                else:
                    total_known += 1
                    if tag != predict_tags[i]:
                        known_errors += 1

        known_error_rate = known_errors / total_known
        unknown_error_rate = unknown_errors / total_unknown
        total_error_rate = (known_errors + unknown_errors) / \
                           (total_known + total_unknown)
        if self.calc_confusion_matrix:
            conf_matrix = confusion_matrix(y_true, y_predicted, normalize='true')
            print(nltk.ConfusionMatrix(y_true, y_predicted))
            return known_error_rate, unknown_error_rate, total_error_rate, conf_matrix
        return known_error_rate, unknown_error_rate, total_error_rate
