import numpy as np
from collections import defaultdict, namedtuple, Counter
import Chu_Liu_Edmonds_algorithm as MST_ALG
from nltk.corpus import dependency_treebank

Arc = namedtuple('Arc', 'head tail weight')
DISTANCE_FEATURE_LEN = 5
NUM_OF_EPOCHS = 2


class TreeParser:
    def __init__(self):
        parsed_sents = np.array(dependency_treebank.parsed_sents())
        # divide the test_set to the last 10% of the corpus
        self.test_set = parsed_sents[(-len(parsed_sents) // 10)+1:]
        self.train_set = parsed_sents[:((-len(parsed_sents) // 10)+1)]

        self.weight_vector = defaultdict()
        self.map_words_to_features = defaultdict(Counter)
        self.map_tags_to_features = defaultdict(Counter)
        self.feature_counter = 0
        self.final_weights = np.array([])
        self.direction_weight = np.zeros(2 * DISTANCE_FEATURE_LEN+1)
        self.distance_weight = np.zeros(DISTANCE_FEATURE_LEN+1)

    def train_model(self):
        c = 0
        for epochs in range(NUM_OF_EPOCHS):
            for i in self.shuffle_randomly_train():
                self.init_feature_function(self.train_set[i].nodes)
                mst = self.get_mst_for_sentence(self.train_set[i].nodes)
                self.update_weights(mst, self.train_set[i].nodes)
                # if c == 1:
                #     break
                if i % 300 == 0:
                    print(f'TRAINING - {c * 300} - iterations')
                    print(self.distance_weight)
                    print(self.direction_weight)
                    c += 1
        for k in self.weight_vector:
            self.weight_vector[k] /= (NUM_OF_EPOCHS * self.train_set.size)
        # arr_w = list(self.weight_vector.values())
        # arr_w.append(0)
        # self.final_weights = np.array(arr_w) / (NUM_OF_EPOCHS * self.train_set.size)
        self.direction_weight /= (NUM_OF_EPOCHS * self.train_set.size)
        self.distance_weight /= (NUM_OF_EPOCHS * self.train_set.size)

    def evaluate(self):
        right_total = np.zeros(self.test_set.size, np.float64)
        c = 1
        for i in range(self.test_set.size):
            self.init_feature_function(self.test_set[i].nodes)
            mst = self.get_mst_for_sentence(self.test_set[i].nodes)
            right_total[i] = self.calc_error(mst, self.test_set[i].nodes)
            if i % 300 == 0:
                print(f'TESTING - {c * 300} - iterations')
                c += 1
        print(np.mean(right_total))
        return np.mean(right_total)

    @staticmethod
    def calc_error(predicted_tree, real_tree):
        predicted_edges = []
        for i in range(1, len(predicted_tree)-1):
            edge = predicted_tree[i]
            predicted_edges.append((edge.head, edge.tail))
        correct = 0
        total = 0
        for word in real_tree.values():
            if word['head'] is not None:
                if (word['head'], word['address']) in predicted_edges:
                    correct += 1
                total += 1
        return correct / total

    #  sent is received as dict of nodes
    def get_mst_for_sentence(self, sent_nodes):
        arcs = []
        for i in range(0, len(sent_nodes)):
            w1 = sent_nodes[i]
            for j in range(1, len(sent_nodes)):
                w2 = sent_nodes[j]
                if i != j:
                    index_feature_word = self.get_words_to_features(w1['word'], w2['word'])
                    index_feature_tag = self.get_tags_to_features(w1['tag'], w2['tag'])
                    weight = self.weight_vector[index_feature_word]+self.weight_vector[index_feature_tag]

                    weight += self.get_direction_feature(w1['address']-w2['address'])
                    if w1['address'] < w2['address']:
                        weight += self.get_distance_feature_2(w2['address']-w1['address'])
                    else:
                        weight += self.get_distance_feature_2(w1['address']-w2['address'])
                    arcs.append(Arc(i, j, -weight))
        return MST_ALG.min_spanning_arborescence_nx(arcs, 0)

    def update_weights(self, current_mst, golden_standard):
        for i in range(len(golden_standard)):
            if golden_standard[i]['head']:
                index_feature_word = self.get_words_to_features(golden_standard[golden_standard[i]['head']]['word'],
                                                                golden_standard[i]['word'])
                index_feature_tag = self.get_tags_to_features(golden_standard[golden_standard[i]['head']]['tag'],
                                                              golden_standard[i]['tag'])
                self.update_weight_vector(index_feature_word, index_feature_tag, 1)
                self.update_direction_feature(golden_standard[i]['head']-golden_standard[i]['address'], 0.5)
                # check the edges are in correct order (left to right)
                if golden_standard[i]['address'] < golden_standard[golden_standard[i]['head']]['address']:
                    self.update_distance_feature_2(golden_standard[golden_standard[i]['head']]['address']-
                                                   golden_standard[i]['address'], 1)
                else:
                    self.update_distance_feature_2(
                        golden_standard[i]['address']-golden_standard[golden_standard[i]['head']]['address'], 1)

        for i in range(1, len(current_mst)+1):
            index_feature_word = self.get_words_to_features(golden_standard[current_mst[i].head]['word'],
                                                            golden_standard[current_mst[i].tail]['word'])
            index_feature_tag = self.get_tags_to_features(golden_standard[current_mst[i].head]['tag'],
                                                          golden_standard[current_mst[i].tail]['tag'])
            self.update_weight_vector(index_feature_word, index_feature_tag, -1)
            # now check if the inferred edge is in right order
            distance = (golden_standard[current_mst[i].head]['address']-
                        golden_standard[current_mst[i].tail]['address'])
            # if current_mst[i].tail < current_mst[i].head:
            self.update_direction_feature(distance, -0.5)
            if distance < 0:
                self.update_distance_feature_2(-distance, -1)
            else:
                self.update_distance_feature_2(distance, -1)

    def update_weight_vector(self, indx1, indx2, value):
        self.weight_vector[indx1] += value
        self.weight_vector[indx2] += value

    def init_feature_function(self, golden_standard):
        for i in range(len(golden_standard)):
            if golden_standard[i]['head']:  # there is no reason to predict the head values because he is the root
                self.init_map_words_to_features(
                    golden_standard[golden_standard[i]['head']]['word'], golden_standard[i]['word'])
                self.init_map_tags_to_features(
                    golden_standard[golden_standard[i]['head']]['tag'], golden_standard[i]['tag'])

    def get_words_to_features(self, w1, w2):
        if w1 in self.map_words_to_features and w2 in self.map_words_to_features[w1]:
            return self.map_words_to_features[w1][w2]
        return 0

    def init_map_words_to_features(self, w1, w2):
        if not (w1 in self.map_words_to_features and w2 in self.map_words_to_features[w1]):
            self.map_words_to_features[w1][w2] = self.feature_counter
            self.weight_vector[self.feature_counter] = 0
            self.feature_counter += 1

    def get_tags_to_features(self, t1, t2):
        if t1 in self.map_tags_to_features and t2 in self.map_tags_to_features[t1]:
            return self.map_tags_to_features[t1][t2]
        return 0

    def init_map_tags_to_features(self, t1, t2):
        if not (t1 in self.map_tags_to_features and t2 in self.map_tags_to_features[t1]):
            self.map_tags_to_features[t1][t2] = self.feature_counter
            self.weight_vector[self.feature_counter] = 0
            self.feature_counter += 1

    def get_direction_feature(self, distance):
        return self.direction_weight[distance] if abs(distance) <= DISTANCE_FEATURE_LEN else 0

    def update_direction_feature(self, distance, value):
        if abs(distance) <= DISTANCE_FEATURE_LEN:
            # if distance < 0:
            #     self.distance_weight[abs(distance + (DISTANCE_FEATURE_LEN // 2))] += value
            # else:
            self.direction_weight[distance] += value

    def get_distance_feature_2(self, distance):
        return self.distance_weight[distance] if distance <= DISTANCE_FEATURE_LEN else 0

    def update_distance_feature_2(self, distance, value):
        if abs(distance) <= DISTANCE_FEATURE_LEN:
            self.distance_weight[distance] += value

    def shuffle_randomly_train(self):
        """
        :return: randomly shuffled range for the train set values
        """
        shuffled_train_set = np.arange(self.train_set.size)
        np.random.shuffle(shuffled_train_set)
        return shuffled_train_set


if __name__ == '__main__':
    tree_parser = TreeParser()
    tree_parser.train_model()
    tree_parser.evaluate()
