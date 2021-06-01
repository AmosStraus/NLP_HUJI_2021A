from collections import Counter, defaultdict
import xlsxwriter
from nltk.corpus import brown


class EX2:
    UNSEEN_TAG = 'NN'
    SMALL_NON_ZERO = 1e-30
    CUTOFF = 2
    PREDICTION_CUTOFF = 5

    def __init__(self):
        """
        (a)
        PREPROCESS & INITIALIZATION
        """
        brown_news_tagged = brown.tagged_sents(categories='news')
        # divide the test_set to the last 10% of the corpus
        self.test_set = brown_news_tagged[(-len(brown_news_tagged) // 10) + 1:]
        self.train_set = brown_news_tagged[:((-len(brown_news_tagged) // 10) + 1)]
        self.word_with_rep = 0
        self.count_words = {}
        self.count_tags = {}
        self.total_yx = 0
        self.b4_trained = False
        self.c4_trained = False
        self.d4_trained = False
        self.e4_trained = False
        self.trans_mle_table = defaultdict(Counter)
        self.emit_mle_table = defaultdict(Counter)
        self.emit_mle_table_pw = defaultdict(Counter)
        self.confusion_mat = defaultdict(Counter)
        self.pseudo_words_set = ["endsining", "endsined", "endsiner", "endsinang", "endsinness",
                                 "endsinhood", "endsinment", "endsinish", "startincaps", "containdigit", "allcaps",
                                 "covfefe"]
        self.real_tags = []

    @staticmethod
    def fix_complex_tags(tag):
        """
        function to fix complex tags removing the redundent parts as instructed
        """
        if '+' in tag:
            tag = tag.split('+')[0]
        if '-' in tag:
            tag = tag.split('-')[0]
        return tag

    def MLE_b_train(self):
        """
        (b)
        Trains the model. emissions e(xi|yi) and transmission q(yi|y_i-1) probabilities
        based on MLE
        :return:
        """
        print("------Train 4b------")
        for sent in self.train_set:
            for word, tag in sent:
                tag = self.fix_complex_tags(tag)
                self.count_words.setdefault(word, {})
                self.count_words[word].setdefault(tag, 0)
                self.count_words[word][tag] += 1
                self.count_tags[tag] = self.count_tags.get(tag, 0) + 1
                self.word_with_rep += 1
        self.b4_trained = True

    def MLE_b_test(self, sentence):
        """
        Returns the tag that appeared most often with the given word
         formally it is max{p(tag|word)} by the MLE: #(tag,word)/#(word)
         Since this is a maximum query, not the exact probability we can omit the division by constant #(word)
        :param sentence:
        :return:
        """
        if not self.b4_trained:
            self.MLE_b_train()
            self.b4_trained = True

        result = []
        for word, tag in sentence:
            if word in self.count_words:
                possible_tags = self.count_words[word]
                MLE_tag = max(possible_tags, key=lambda k: possible_tags[k])
                result.append(MLE_tag)  # the tag that appeared most often with the given word
            else:  # word unseen in training_set
                result.append(EX2.UNSEEN_TAG)
        return result

    def MLE_4c_train(self):
        """
        (c)
        Trains the model. emissions e(xi|yi) and transmission q(yi|y_i-1) probabilities
        based on MLE
        """
        print("------Train 4c------")
        # train trans_mle_table
        for sent in self.train_set:
            last_tag = '***'
            for word, tag in sent:
                tag = self.fix_complex_tags(tag)
                self.trans_mle_table[last_tag][tag] += 1
                last_tag = tag
            self.trans_mle_table[last_tag]['STOP'] += 1
        # train x_mle aka e_mle(x|y)
        for sent in self.train_set:
            last_tag = ''
            for word, tag in sent:
                tag = self.fix_complex_tags(tag)
                self.emit_mle_table[tag][word] += 1
                last_tag = tag
            self.emit_mle_table[last_tag]['STOP'] += 1
        self.c4_trained = True

    def e_mle(self, x, y):
        """
        calculates the emission MLE by the formula we have seen
        """
        return self.emit_mle_table[y][x] / sum(self.emit_mle_table[y].values())

    def e_mle_nz(self, x, y):
        """
        calculates the emission MLE by the formula we have seen
        but avoid zero probabilities
        """
        if self.emit_mle_table[y][x] == 0:
            return self.SMALL_NON_ZERO
        return self.emit_mle_table[y][x] / sum(self.emit_mle_table[y].values())

    def t_mle(self, y, y_prev):
        """
         calculates the transmission MLE by the formula we have seen
        """

        return self.trans_mle_table[y_prev][y] / sum(self.trans_mle_table[y_prev].values())

    def viterbi(self, sent, q, e):
        """
        implementing the viterbi algorithm
        """
        n = len(sent)
        # initialize
        table = defaultdict(Counter)
        table[0]['***'] = 1  # all words begin with this token

        # define all possible tags in our case S_k=S for all k in range
        S = [s for s in self.trans_mle_table if s != '***']

        # backtracking the maximal tags.
        backtrack = {}

        # iteration start from 1 because the start token is *
        for k in range(1, n + 1):
            # keeps track of all the transitions in the k's step
            backtrack[k] = {}
            for y in S:
                x_k = sent[k - 1][0]
                if k == 1:
                    # no previous tags
                    table[k][y] = table[k - 1]['***'] * q(y, '***') * e(x_k, y)
                    backtrack[k][y] = '***'  # just for completeness
                else:
                    # search for the max previous tag
                    tag_max = EX2.UNSEEN_TAG
                    maxvalue = 0
                    for y_prev in S:
                        temp = table[k - 1][y_prev] * q(y, y_prev) * e(x_k, y)
                        if temp > maxvalue:
                            maxvalue = temp
                            tag_max = y_prev
                    table[k][y] = maxvalue
                    backtrack[k][y] = tag_max

        last_tag = EX2.UNSEEN_TAG
        last_maxvalue = 0
        for v in S:
            temp2 = table[n][v] * q('STOP', v)
            if temp2 > last_maxvalue:
                last_maxvalue = temp2
                last_tag = v
        y_i_plus1 = last_tag
        result = [last_tag]

        for i in range(n - 1, 0, -1):
            yi = backtrack[i + 1][y_i_plus1]
            result.insert(0, yi)
            y_i_plus1 = yi
        return result

    def HMM_4c_test(self, sent):
        """
        utilizing the viterbi algorithm to calculate the bigram HMM
        """
        if not self.b4_trained:
            self.MLE_b_train()
        if not self.c4_trained:
            self.MLE_4c_train()

        return self.viterbi(sent, self.t_mle, self.e_mle)

    def HMM_4c_test_nz(self, sent):
        """
        utilizing the viterbi algorithm to calculate the bigram HMM
        """
        if not self.b4_trained:
            self.MLE_b_train()
        if not self.c4_trained:
            self.MLE_4c_train()

        return self.viterbi(sent, self.t_mle, self.e_mle_nz)

    def e_mle_1smooth(self, x, y):
        """
        calculates the emission MLE by the formula we have seen, with add 1 smoothing
        """
        return (self.emit_mle_table[y][x] + 1) / (len(self.count_words) + sum(self.emit_mle_table[y].values()))

    def e_mle_1smooth_nz(self, x, y):
        """
        calculates the emission MLE by the formula we have seen, with add 1 smoothing
        """
        a = 0
        if self.emit_mle_table[y][x] == 0:
            a = self.SMALL_NON_ZERO
        return (a + self.emit_mle_table[y][x] + 1) / (len(self.count_words) + sum(self.emit_mle_table[y].values()))

    def smoothed_4d_test(self, sent):
        """
        (d)
        Trains and tests the model. emissions e(xi|yi) and transmission q(yi|y_i-1) probabilities
        using the add 1 smoothing for emission, calculates the new probabilities, using the viterbi algorithm
        :param sent:
        :return:
        """
        if not self.b4_trained:
            self.MLE_b_train()
        if not self.c4_trained:
            self.MLE_4c_train()
        self.total_yx = sum([sum(self.emit_mle_table[y].values()) for y in self.emit_mle_table])
        return self.viterbi(sent, self.t_mle, self.e_mle_1smooth)

    def accuracy(self, model):
        """
        accuracy measures the number of correctly tagged words divided by the test set size
        model - the model we are testing
        """
        counter_correct, counter_total, counter_unseen, unseen_correct = 0, 0, 0, 0
        size = sum([len(self.test_set[i]) for i in range(len(self.test_set))])
        for j, test_sent in enumerate(self.test_set):
            prediction = []
            if model not in ['b', 'c', 'd', 'e1', 'e2', 'e2_nz']:
                return
            if model == 'b':
                prediction = self.MLE_b_test(test_sent)
            if model == 'c':
                prediction = self.HMM_4c_test(test_sent)
            if model == 'c_nz':
                prediction = self.HMM_4c_test_nz(test_sent)
            if model == 'd':
                prediction = self.smoothed_4d_test(test_sent)
            if model == 'e1':
                prediction = self.e_pw_test(test_sent)
            if model == 'e2':
                prediction = self.e_smoothed_pw_test(test_sent)
            if model == 'e2_nz':
                prediction = self.e_smoothed_pw_test_nz(test_sent)
            for i, (word, real_tag) in enumerate(test_sent):
                real_tag = self.fix_complex_tags(real_tag)
                counter_total += 1
                self.confusion_mat[real_tag][prediction[i]] += 1
                if prediction[i] == real_tag:
                    counter_correct += 1
                    if word not in self.count_words:
                        unseen_correct += 1
                        # print(counter_total, 'out of', size)

                if word not in self.count_words:
                    counter_unseen += 1

        return [
            counter_correct / counter_total,
            (counter_correct - unseen_correct) / (counter_total - counter_unseen),
            unseen_correct / counter_unseen
        ]

    def error_rate(self, model):
        """
        error rate is 1-accuracy by definition
        :param model:
        :return:
        """
        accuracy = self.accuracy(model)
        if accuracy:
            print(f'{model} error rate:')
            print('Total  error rate', 1 - accuracy[0])
            print('Seen   error rate', 1 - accuracy[1])
            print('Unseen error rate', 1 - accuracy[2])

    def e_mle_pw(self, x, y):
        """
        calculates the emission MLE by the formula we have seen
        """
        if x not in self.count_words or sum(self.count_words[x].values()) < self.PREDICTION_CUTOFF:
            x = self.replace_pw(x)
        return (self.emit_mle_table_pw[y][x]) / sum(self.emit_mle_table_pw[y].values())

    def e_mle_pw_smoothed(self, x, y):
        """
        calculates the emission MLE by the formula we have seen, with add 1 smoothing
        """
        if x not in self.count_words or sum(self.count_words[x].values()) < self.PREDICTION_CUTOFF:
            x = self.replace_pw(x)
        return (self.emit_mle_table_pw[y][x] + 1) / (len(self.count_words) + sum(self.emit_mle_table_pw[y].values()))

    def e_mle_pw_smoothed_nz(self, x, y):
        """
        calculates the emission MLE by the formula we have seen, with add 1 smoothing
        """
        a = 0
        if self.emit_mle_table[y][x] == 0:
            a = self.SMALL_NON_ZERO
        if x not in self.count_words or sum(self.count_words[x].values()) < self.PREDICTION_CUTOFF:
            x = self.replace_pw(x)
        return (a + self.emit_mle_table_pw[y][x] + 1) / (
                len(self.count_words) + sum(self.emit_mle_table_pw[y].values()))

    def replace_pw(self, word):
        if type(word) == tuple:
            word = word[0]

        if word.endswith('ang'):
            return "endsinANG"
        elif word.endswith('ness'):
            return "endsinNESS"
        elif word.endswith('hood'):
            return "endsinHOOD"
        elif word.endswith('ment'):
            return "endsinMENT"
        elif word.endswith('ish'):
            return "endsinISH"
        elif 'be' in word:
            return "containsBE"
        elif word.isupper():
            return "allCaps"
        elif word.endswith('ing'):
            return "endsinING"
        elif word.endswith('ed'):
            return "endsinED"
        elif word.endswith('er'):
            return "endsinER"
        elif any(c.isdigit() for c in word):
            return "containsDigit"
        elif word.endswith('it'):
            return "endsinIT"
        elif word.endswith('ly'):
            return "endsinLY"
        elif word[0].isupper():
            return "initCap"
        else:
            return "defaultPW"

    def mle_4e_train(self):
        if not self.b4_trained:
            self.MLE_b_train()
        if not self.c4_trained:
            self.MLE_4c_train()

        for tag in self.emit_mle_table:
            for word in self.emit_mle_table[tag]:
                if self.emit_mle_table[tag][word] < self.CUTOFF:
                    self.emit_mle_table_pw[tag][self.replace_pw(word)] += self.emit_mle_table[tag][word]
                    del self.emit_mle_table_pw[tag][word]
                else:
                    self.emit_mle_table_pw[tag][word] = self.emit_mle_table[tag][word]
        self.e4_trained = True

    def e_pw_test(self, sent):
        if not self.e4_trained:
            self.mle_4e_train()
        return self.viterbi(sent, self.t_mle, self.e_mle_pw)

    def e_smoothed_pw_test(self, sent):
        if not self.b4_trained:
            self.MLE_b_train()
        if not self.c4_trained:
            self.MLE_4c_train()
        if not self.e4_trained:
            self.mle_4e_train()

        return self.viterbi(sent, self.t_mle, self.e_mle_pw_smoothed)

    def e_smoothed_pw_test_nz(self, sent):
        if not self.b4_trained:
            self.MLE_b_train()
        if not self.c4_trained:
            self.MLE_4c_train()
        if not self.e4_trained:
            self.mle_4e_train()

        return self.viterbi(sent, self.t_mle, self.e_mle_pw_smoothed_nz)

    # def confusion_matrix(self, test_sent):  # ugly but works
    #     prediction = self.e_smoothed_pw_test(test_sent)
    #     K = len(self.emit_mle_table_pw.keys())
    #     work_tags = list(self.emit_mle_table_pw.keys())
    #     print(work_tags)
    #     cm = np.zeros((K, K))  # 98 * 98 matrix
    #     for i in range(K):
    #         true_tag = work_tags[i]  # rows
    #         for j in range(K):
    #             predict_tag = work_tags[j]  # cols
    #             for p in range(len(prediction)):  # for i in prediction (len 462)
    #                 for k in range(len(self.test_set[p])):
    #                     if prediction[p] == predict_tag and self.test_set[p][k][1] == true_tag:
    #                         cm[i][j] += 1
    #     df_cm = pd.DataFrame(cm, index=[i for i in work_tags], columns=[i for i in work_tags])
    #     pd.DataFrame(df_cm).to_csv("cm.csv")
    #
    #     print(df_cm)

    def mat_to_excel(self, model_name):
        workbook = xlsxwriter.Workbook(f'D:\\2021a_HUJI\\NLP\\NLP_ex2\\{model_name}_confusion_mat.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.write(0, 0, 'truth/prediction')
        for i, y in enumerate(self.confusion_mat):
            worksheet.write(0, 1 + i, y)
            worksheet.write(1 + i, 0, y)
        i = 1
        for row in self.confusion_mat:
            j = 1
            for col in self.confusion_mat:
                worksheet.write(i, j, self.confusion_mat[row][col])
                j += 1
            i += 1
        workbook.close()


""" Trains all models and computes the required error-rate/accuract values"""


def run_all():
    classifier = EX2()
    classifier.error_rate('b')
    classifier.error_rate('c')
    classifier.error_rate('d')
    classifier.error_rate('e1')
    classifier.error_rate('e2')
    classifier.mat_to_excel('e2')


def tester():
    classifier = EX2()
    test_sentence = [('Last', 'AP'), ('week', 'NN'), ('Federal', 'JJ-TL'), ('District', 'NN-TL'), ('Judge', 'NN-TL'),
                     ('William', 'NP'), ('A.', 'NP'), ('Bootle', 'NP'), ('ordered', 'VBD'), ('the', 'AT'),
                     ('university', 'NN'),
                     ('to', 'TO'), ('admit', 'VB'), ('immediately', 'RB'), ('a', 'AT'), ('``', '``'),
                     ('qualified', 'VBN'),
                     ("''", "''"), ('Negro', 'NP'), ('boy', 'NN'), ('and', 'CC'), ('girl', 'NN'), ('.', '.')]

    c = classifier.HMM_4c_test_nz(test_sentence)
    b = classifier.smoothed_4d_test(test_sentence)
    e2 = classifier.e_smoothed_pw_test(test_sentence)
    e3 = classifier.e_smoothed_pw_test_nz(test_sentence)
    e1 = classifier.e_pw_test(test_sentence)
    test_sent_pw = [classifier.replace_pw(x[0]) for x in test_sentence if x[0] not in classifier.count_words]
    print(classifier.emit_mle_table_pw)
    print(test_sent_pw)
    truth = [classifier.fix_complex_tags(t) for w, t in test_sentence]
    print(classifier.emit_mle_table['RB']['immediately'])
    print(classifier.emit_mle_table_pw['RB']['immediately'])
    print(classifier.emit_mle_table['VB']['admit'])
    print(classifier.e_mle_1smooth('VB', 'admit'))
    print(c)
    print(b)
    print(e1)
    print(e2)
    print(e3)
    print(truth)
    print(sum([truth[i] == e1[i] for i in range(len(truth))]), 'out of', len(truth))
    print(sum([truth[i] == e3[i] for i in range(len(truth))]), 'out of', len(truth))

    print('SECOND SENTENCE')
    test_sentence2 = classifier.test_set[-380]
    truth = [classifier.fix_complex_tags(t) for w, t in test_sentence2]
    c_nz = classifier.HMM_4c_test_nz(test_sentence2)
    e1 = classifier.e_pw_test(test_sentence2)
    e2 = classifier.e_smoothed_pw_test(test_sentence2)
    e3 = classifier.e_smoothed_pw_test_nz(test_sentence2)
    print(test_sentence2)
    print(c_nz)
    print(e1)
    print(e2)
    print(e3)
    print(truth)
    print(sum([truth[i] == e1[i] for i in range(len(truth))]), 'out of', len(truth))
    print(sum([truth[i] == e3[i] for i in range(len(truth))]), 'out of', len(truth))
    print("confusion")
    print(len(classifier.count_words))
    print(classifier.replace_pw('detested'))
    print(classifier.replace_pw('being'))
    print(classifier.replace_pw('Belgians'))
    print(classifier.emit_mle_table_pw['VBN']['detested'])
    print(classifier.emit_mle_table_pw['VBN'][classifier.replace_pw('detested')])
    print(classifier.emit_mle_table_pw['JJ']['detested'])
    print(classifier.emit_mle_table_pw['JJ'][classifier.replace_pw('detested')])
    print(classifier.trans_mle_table['AT']['JJ'])
    print(classifier.trans_mle_table['AT']['VBN'])

    classifier.error_rate('b')
    classifier.mat_to_excel('b')


if __name__ == '__main__':
    # tester()
    run_all()
