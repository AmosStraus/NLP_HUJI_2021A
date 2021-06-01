import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import data_loader
import pickle
import matplotlib.pyplot as plt

# ------------------------------------------- Constants ----------------------------------------

LogLinear_epochs = 20
LSTM_epochs = 4
SEQ_LEN = 18
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=True):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):  # todo
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    all_vec = []
    for word in sent.text:
        if word in word_to_vec:
            all_vec.append(word_to_vec[word])
    if len(all_vec) == 0:
        w2v_average = np.zeros(embedding_dim)
    else:
        all_vec = np.array(all_vec)
        w2v_average = np.mean(all_vec, axis=0)
    return w2v_average


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    res = np.zeros(size)
    if size > ind >= 0:
        res[ind] = 1
    return res


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    res = np.zeros(len(word_to_ind))
    for word in sent.text:
        res[word_to_ind[word]] += 1
    res /= len(sent.text)
    return res


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    res = {}
    i = 0
    for word in words_list:
        if word not in res:
            res[word] = i
            i += 1
    return res


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    first_ten_embedding = []
    for i, word in enumerate(sent.text):
        if i == SEQ_LEN:
            break
        if word in word_to_vec:
            first_ten_embedding.append(word_to_vec[word])
        else:
            first_ten_embedding.append(np.zeros(embedding_dim, dtype=float))

    if len(sent.text) < SEQ_LEN:
        for j in range(SEQ_LEN-len(sent.text)):
            first_ten_embedding.append(np.zeros(embedding_dim, dtype=float))

    return np.array(first_ten_embedding)


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim=100, n_layers=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        # MULTIPLY BY 2 FOR CONCATENATING #
        self.linear_out = nn.Linear(in_features=hidden_dim * 2, out_features=1)

    def forward(self, text):
        lstm_out, hidden = self.lstm(text)
        # Take only hidden layers and input them to the linear layer #
        return self.linear_out(lstm_out[:, -1, :])

    def predict(self, text):
        lstm_out, hidden = self.lstm(text)
        res = self.linear_out(lstm_out[:, -1, :])
        res = nn.Sigmoid()(res)
        return torch.round(res).type(torch.FloatTensor).transpose(0, 1)

    def single_predict(self, x):
        lstm_out, hidden = self.lstm(x.resize_(1, x.shape[0], x.shape[1]))
        res = self.linear_out(lstm_out[:, -1, :])
        res = nn.Sigmoid()(res)
        return torch.round(res)


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=embedding_dim, out_features=1)

    def forward(self, x):
        h1 = self.linear1(x)
        return h1

    def predict(self, x):
        res = self.linear1(x)
        res = nn.Sigmoid()(res)
        return torch.round(res).type(torch.FloatTensor).transpose(0, 1)

    def single_predict(self, x):
        res = self.linear1(x)
        res = nn.Sigmoid()(res)
        return torch.round(res)


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """

    if not len(preds) == len(y):
        print("WRONG DIMENSIONS IN BINARY ACCURACY GOT {} {}".format(len(preds), len(y)))
    else:
        return np.mean(preds.detach().numpy() == y.numpy())


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    loss_total = 0
    current_accuracy = 0

    for x, y in data_iterator:
        x = x.to(get_available_device())
        y = y.to(get_available_device())
        optimizer.zero_grad()
        pred = model(x.type(torch.FloatTensor)).transpose(0, 1).squeeze(0)
        loss = criterion(pred, y)
        loss_total += loss
        loss.backward()
        optimizer.step()
        current_accuracy += binary_accuracy(torch.round(nn.Sigmoid()(pred)), y)

    return (current_accuracy / len(data_iterator)), (loss_total / len(data_iterator))


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    loss_total = 0
    current_accuracy = 0
    for x, y in data_iterator:
        x = x.to(get_available_device())
        y = y.to(get_available_device())
        pred = model(x.type(torch.FloatTensor)).transpose(0, 1).squeeze(0)
        loss = criterion(pred, y)
        loss_total += loss
        loss.backward()
        current_accuracy += binary_accuracy(torch.round(nn.Sigmoid()(pred)), y)

    return (current_accuracy / len(data_iterator)), (loss_total / len(data_iterator))


# TODO: AMOS ADDED THIS FUNCTION. USE IT AS YOU WANT INCLUDE THE CHANGE IN README
def get_rare_and_polarity_prediction(model, data_manager, dataset):
    """
    ADDED FUNCTION TO HANDLE THE SPECIAL SUBSETS OF THE TEST SET
    :param model: model we train
    :param data_manager: the data manager
    :param dataset: the dataset
    :return: the accuracy of the model on the special subsets (RARE, NEGATED)
    """
    rare_words_ind = data_loader.get_rare_words_examples(dataset.get_test_set(), dataset)
    negated_polarity_ind = data_loader.get_negated_polarity_examples(dataset.get_test_set())
    i = 0
    rare_predictions = []
    negated_predictions = []
    for x, y in (data_manager.torch_iterators[TEST]):
        for j in range(len(x)):
            if i in rare_words_ind:
                rare_predictions.append(model.single_predict(x[j].type(torch.FloatTensor)) == y[j])
            if i in negated_polarity_ind:
                negated_predictions.append(model.single_predict(x[j].type(torch.FloatTensor)) == y[j])
            i += 1
    print("{} out of {}".format(sum(rare_predictions), len(rare_predictions)))
    print("{} out of {}".format(sum(negated_predictions), len(negated_predictions)))
    return np.mean(rare_predictions), np.mean(negated_predictions)


"""
    I WROTE ANOTHER FUNCTION INSTEAD THIS ONE, IT SEEMS REDUNDANT
"""


# def get_predictions_for_data(model, data_iter):
#     """
#
#     This function should iterate over all batches of examples from data_iter and return all of the models
#     predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
#     same order of the examples returned by data_iter.
#     :param model: one of the models you implemented in the exercise
#     :param data_iter: torch iterator as given by the DataManager
#     :return:
#     """
#     predictions = []
#     for x, y in data_iter:
#         pred = model.predict(x.type(torch.FloatTensor)).transpose(0, 1).squeeze(0)
#         predictions.append(pred)
#     return np.array(predictions)


def train_model(model, data_manager, n_epochs, lr, weight_decay=0., model_name="LogLinear_1"):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model_name: The model name for saving the data and plots
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss().to(get_available_device())
    TRAIN_epochs_accuracies_losses = []
    VALIDATION_epochs_accuracies_losses = []
    for epoch in range(n_epochs):
        #
        # TRAIN #
        train_iter = data_manager.get_torch_iterator(data_subset=TRAIN)
        accuracy_train, loss_train = train_epoch(model, train_iter, optimizer, criterion)
        print(f"epoch {epoch}")
        print("TRAIN \n\taccuracy {} - loss {}".format(accuracy_train, loss_train))
        TRAIN_epochs_accuracies_losses.append((epoch, accuracy_train, loss_train))
        #
        # EVALUATE / VALIDATE #
        validation_iter = data_manager.get_torch_iterator(data_subset=VAL)
        accuracy_validation, loss_validation = evaluate(model, validation_iter, criterion)
        print("VALIDATION \n\taccuracy {} - loss {}".format(accuracy_validation, loss_validation))
        VALIDATION_epochs_accuracies_losses.append((epoch, accuracy_validation, loss_validation))
    #
    # SAVE MODEL PARAMETERS AND RESULTS ON TRAIN/VALIDATION DATA
    save_model(model, "{}_MODEL_PARAMS".format(model_name), n_epochs, optimizer)
    save_pickle(TRAIN_epochs_accuracies_losses, "{}(_TRAIN)".format(model_name))
    save_pickle(VALIDATION_epochs_accuracies_losses, "{}(_VALIDATION)".format(model_name))


def train_and_test_helper(model_name):
    dm = None
    model = None
    # TRAIN
    if model_name == "LogLinear_1":
        dm = DataManager(data_type=ONEHOT_AVERAGE, batch_size=64)
        model = LogLinear(dm.get_input_shape()[0]).to(get_available_device())
        train_model(model, dm, LogLinear_epochs, 0.01, 0.0001, "LogLinear_1")
    elif model_name == "LogLinear_W2V":
        dm = DataManager(data_type=W2V_AVERAGE, embedding_dim=W2V_EMBEDDING_DIM, batch_size=64)
        model = LogLinear(dm.get_input_shape()[0]).to(get_available_device())
        train_model(model, dm, LogLinear_epochs, 0.01, 0.0001, "LogLinear_W2V")
    elif model_name == "LSTM":
        dm = DataManager(data_type=W2V_SEQUENCE, embedding_dim=W2V_EMBEDDING_DIM, batch_size=64)
        model = LSTM(embedding_dim=W2V_EMBEDDING_DIM).to(get_available_device())
        train_model(model, dm, LSTM_epochs, 0.001, 0.0001, "LSTM")

    # TEST #
    TEST_RESULTS = []
    test_iter = dm.get_torch_iterator(data_subset=TEST)
    accuracy_test, loss_test = evaluate(model, test_iter, nn.BCEWithLogitsLoss())
    rare_results, negated_results = get_rare_and_polarity_prediction(
        model, dm, dm.sentiment_dataset)

    print(f"{model_name} RESULTS:")
    print("TEST Overall: accuracy {} - loss {}".format(accuracy_test, loss_test))
    print("RARE: accuracy {}".format(rare_results))
    print("NEGATED: accuracy {} ".format(negated_results))
    TEST_RESULTS.append(("TEST_RESULTS", accuracy_test, loss_test))
    TEST_RESULTS.append(("RARE_RESULTS", rare_results))
    TEST_RESULTS.append(("NEGATED_RESULTS", negated_results))

    save_pickle(TEST_RESULTS, f"{model_name}(_TEST)")


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    train_and_test_helper("LogLinear_1")


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    train_and_test_helper("LogLinear_W2V")


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    train_and_test_helper("LSTM")


def plot_results(model_name):
    """
    Plots the result of the model on the TRAIN and VALIDATION sets
    :param model_name: string name
    """
    epochs = LogLinear_epochs
    if model_name is "LSTM":
        epochs = LSTM_epochs
    train_loss = []
    train_acc = []

    train_res = load_pickle(f"{model_name}(_TRAIN)")
    for e1, a1, l1 in train_res:
        train_loss.append(l1)
        train_acc.append(a1)

    val_res = load_pickle(f"{model_name}(_VALIDATION)")
    val_loss = []
    val_acc = []
    for e2, a2, l2 in val_res:
        val_loss.append(l2)
        val_acc.append(a2)

    fig1, ax1 = plt.subplots()
    ax1.set(xlabel='epochs', ylabel='accuracy',
            title=f'{model_name} ACCURACY')
    ax1.plot(np.arange(1, epochs+1, 1), train_acc, '-b', label='train accuracy')
    ax1.plot(np.arange(1, epochs+1, 1), val_acc, '-g', label='validation accuracy')
    plt.legend()
    plt.savefig(f'{model_name} ACCURACY.png')
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.set(xlabel='epochs', ylabel='loss',
            title=f'{model_name} LOSS')
    ax2.plot(np.arange(1, epochs+1, 1), train_loss, '-b', label='train loss')
    ax2.plot(np.arange(1, epochs+1, 1), val_loss, '-g', label='validation loss')
    plt.legend()
    plt.savefig(f'{model_name} LOSS.png')
    plt.show()


def get_model_parameters(model_name):
    """
    Loads the TRAINED model parameters.
    :param model_name: one of the 3 models we trained
    :return: tuple (the model, the optimizer and the epoch)
    """
    if model_name == "LogLinear_1":
        dm1 = DataManager(data_type=ONEHOT_AVERAGE, batch_size=64)
        log_linear_model = LogLinear(dm1.get_input_shape()[0])
        m, o, e = load(log_linear_model, "LogLinear_1(_MODEL_PARAMS",
                       torch.optim.Adam(params=log_linear_model.parameters(), lr=0.01, weight_decay=0.0001))
        return m, o, e

    elif model_name == "LogLinear_W2V":
        dm2 = DataManager(data_type=W2V_AVERAGE, embedding_dim=W2V_EMBEDDING_DIM, batch_size=64)
        log_linear_model_w2v = LogLinear(dm2.get_input_shape()[0])
        m2, o2, e2 = load(log_linear_model_w2v, "LogLinear_W2V_MODEL_PARAMS",
                          torch.optim.Adam(params=log_linear_model_w2v.parameters(), lr=0.01, weight_decay=0.0001))
        return m2, o2, e2

    else:
        dm3 = DataManager(data_type=W2V_SEQUENCE, embedding_dim=W2V_EMBEDDING_DIM, batch_size=64)
        lstm = LSTM(W2V_EMBEDDING_DIM)
        m3, o3, e3 = load(lstm, "LSTM_MODEL_PARAMS",
                          torch.optim.Adam(params=lstm.parameters(), lr=0.001, weight_decay=0.0001))
        return m3, o3, e3


if __name__ == '__main__':
    # train_log_linear_with_one_hot()
    # train_log_linear_with_w2v()
    # train_lstm_with_w2v()

    # plot_results('LogLinear_1')
    # plot_results('LogLinear_W2V')
    plot_results('LSTM')
    # val_res = load_pickle("LSTM(_TEST)")
    # print(val_res)
