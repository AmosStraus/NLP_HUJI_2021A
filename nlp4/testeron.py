import torch
import numpy as np
from torch import nn


class SimpleNNWithLogits(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()  # Important to call Modules constructor!!
        self.linear1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, x):
        h1 = self.linear1(x)
        h1 = nn.ReLU()(h1)
        h1 = self.dropout1(h1)
        h2 = self.linear2(h1)
        h2 = nn.ReLU()(h2)
        h2 = self.dropout2(h2)
        out = self.linear3(h2)
        return out

    def optimize(self, x_tensor, y_tensor):
        op = torch.optim.Adam(params=self.parameters(), lr=1e-3, weight_decay=0)
        for i in range(2000):
            op.zero_grad()
            pred = NN_logits(x_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            op.step()
            if i % 100 == 0:
                print(list(self.named_parameters()))
                print("step: {}. loss: {}".format(i, loss))


if __name__ == '__main__':
    NN_logits = SimpleNNWithLogits(input_dim=10, hidden_dim=20)
    print(list(NN_logits.named_parameters())[:1][:10])

    # try to predict if the sum of even entries in a vector is larger then the odd entries.
    x = np.random.randint(0, 2, (1000, 10))
    y = (np.sum(x[:, ::2], axis=1, keepdims=True) - np.sum(x[:, 1::2], axis=1, keepdims=True)) > 0

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    pred = NN_logits(x_tensor)
    print("first 10 random predictions: \n", pred[:10])
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(pred, y_tensor)
    print("loss with random weights: \n", loss)

    # now we can calculate the gradient with respect to the loss!
    loss.backward()
    first_bias = list(NN_logits.parameters())[1]
    print("gradient for the first layer's bias weights: \n", first_bias.grad)

    # FOR THE GRADIENT DESCENT STEPS
    # optimizer = torch.optim.Adam(params=NN_logits.parameters(), lr=1e-3, weight_decay=0)
    # nn.BCEWithLogitsLoss()
    # for i in range(1000):
    #     optimizer.zero_grad()
    #     pred = NN_logits(x_tensor)
    #     loss = criterion(pred, y_tensor)
    #     loss.backward()
    #     optimizer.step()
    #     if i % 100 == 0:
    #         print("loss at step {}: {}".format(i, loss))

    # NN_logits.train()
    # print("1st pred in train: ", NN_logits(x_tensor[:1]))
    # print("2nd pred in train: ", NN_logits(x_tensor[:1]))
    # NN_logits.eval()
    # print("1st pred in test: ", NN_logits(x_tensor[:1]))
    # print("2nd pred in test: ", NN_logits(x_tensor[:1]))

    new_x = np.random.randint(0, 2, (1000, 10))
    new_y = (np.sum(new_x[:, ::2], axis=1, keepdims=True) - np.sum(new_x[:, 1::2], axis=1, keepdims=True)) > 0

    new_x_tensor = torch.tensor(new_x, dtype=torch.float32)
    new_y_tensor = torch.tensor(new_y, dtype=torch.float32)

    logits = NN_logits(x_tensor)
    print(logits[:10])
    # now we need the probabilities to have a prediction
    probs = nn.Sigmoid()(logits)
    print(probs[:10])
    new_pred = (probs > 0.5).numpy()
    print(new_pred.shape)
    print(new_pred.shape)
    print("test accuracy: ", np.mean(new_pred == new_y))

    NN_logits.optimize(x_tensor, y_tensor)
    logits = NN_logits(x_tensor)
    # now we need the probabilities to have a prediction
    probs = nn.Sigmoid()(logits)
    new_pred = (probs > 0.5).numpy()
    print("test accuracy: ", np.mean(new_pred == new_y))

    # new_x_2 = np.random.randint(0, 2, (1000, 10))
    # new_x_tensor_2 = torch.tensor(x, dtype=torch.float32)
