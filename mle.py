class Mle:
    def __init__(self, train_set):
        self.model = dict()
        self.train_set = train_set

    def train(self):
        for sentence in self.train_set:
            for word, tag in sentence:
                if word not in self.model:
                    self.model[word] = {tag: 1}
                else:
                    if tag not in self.model[word]:
                        self.model[word][tag] = 1
                    else:
                        self.model[word][tag] += 1
        for word in self.model:
            self.model[word] = max(self.model[word], key=self.model[word].get)

    def calculate_error_rate(self, test_set):
        unknown_errors = 0
        known_errors = 0
        total_known = 0
        total_unknown = 0
        for sentence in test_set:
            for word, tag in sentence:
                if word not in self.model:
                    total_unknown += 1
                    if tag != 'NN':
                        unknown_errors += 1
                else:
                    total_known += 1
                    if tag != self.model[word]:
                        known_errors += 1

        known_error_rate = known_errors / total_known
        unknown_error_rate = unknown_errors / total_unknown
        total_error_rate = (known_errors + unknown_errors) / \
                           (total_known + total_unknown)
        return known_error_rate, unknown_error_rate, total_error_rate
