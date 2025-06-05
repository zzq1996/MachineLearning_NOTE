import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import load_diabetes


class LinearRegressionModel():
    def __int__(self):
        pass

    def prepare_data(self):
        data = load_diabetes().data
        target = load_diabetes().target
        X, y = shuffle(data, target, random_state=42)
        X = X.astype(np.float32)
        y = y.reshape((-1, 1))
        # axis的值为0,这意味着要沿着第一个轴(即行)连接这些数组。
        # 如果将axis设置为负数，则沿着最后一个轴(即列)连接这些数组。
        data = np.concatenate((X, y), axis=1)
        return data

    def initialize_params(self, dims):
        w = np.zeros((dims, 1))
        b = 0
        return w, b

    def linear_loss(self, X, y, w, b):
        num_train = X.shape[0]
        num_feature = X.shape[1]

        y_hat = np.dot(X, w) + b
        loss = np.sum((y_hat - y) ** 2) / num_train
        dw = np.dot(X.T, (y_hat - y)) / num_train
        db = np.sum((y_hat - y)) / num_train
        return y_hat, loss, dw, db

    def linear_train(self, X, y, learning_rate, epochs):
        w, b = self.initialize_params(X.shape[1])
        for i in range(1, epochs):
            y_hat, loss, dw, db = self.linear_loss(X, y, w, b)
            w += -learning_rate * dw
            b += -learning_rate * db
            if i % 10000 == 0:
                print('epoch %d loss %f' % (i, loss))
            params = {
                'w': w,
                'b': b
            }
            grads = {
                'dw': dw,
                'db': db
            }
        return loss, params, grads

    def predict(self, X, params):
        w = params['w']
        b = params['b']
        y_pred = np.dot(X, w) + b
        return y_pred

    def linear_cross_validation(self, data, k, randomize=True):
        if randomize:
            data = list(data)
            shuffle(data)

        slices = [data[i::k] for i in range(k)]

        for i in range(k):
            validation = slices[i]
            train = [data
                     for s in slices if s is not validation for data in s]
            train = np.array(train)
            validation = np.array(validation)
            yield train, validation


if __name__ == '__main__':
    lr = LinearRegressionModel()
    data = lr.prepare_data()
    for train, validation in lr.linear_cross_validation(data, 5):
        X_train = train[:, :10]
        y_train = train[:, -1].reshape((-1, 1))
        X_valid = validation[:, :10]
        y_valid = validation[:, -1].reshape((-1, 1))
        loss5 = []
        loss, params, grads = lr.linear_train(X_train, y_train, 0.001, 100000)
        loss5.append(loss)
        score = np.mean(loss5)
        print('five kold cross validation score is', score)
        y_pred = lr.predict(X_valid, params)
        valid_score = np.sum(((y_pred - y_valid) ** 2)) / len(X_valid)
        print('valid score is', valid_score)
