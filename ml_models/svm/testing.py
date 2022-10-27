import pandas as pd
import numpy as np
<<<<<<< HEAD
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class svc:
    def __init__(self, sigma, kernel: str = 'rbf', C: int = 1, n_iterations=1000, lr=0.01, momentum=0.001):
        self.kernel = kernel
        self.C = C
        self.n_iterations = n_iterations
        self.lr = lr
        self.momentum = momentum
        self.sigma = sigma

    def fit(self, X, y):

        # initialize alpha
        alpha = np.zeros(X.shape[0])
        # pre-compute kernel matrix -- since we don't need to calculate kernel upon each epoch
        kernel_matrix = np.zeros([X.shape[0], X.shape[0]])
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if self.kernel == 'linear':
                    kernel_matrix[i, j] = np.dot(X[i, :], X[:, j])
                elif self.kernel == 'rbf':
                    kernel_matrix[i, j] = rbf(X[i], X[j], self.sigma)

        for i in range(self.n_iterations):
            # calculate gradient
            gradient = -1 + y * np.matmul(kernel_matrix, alpha * y) + self.C * np.dot(alpha, y) * y
            # update alphas
            alpha -= self.lr * gradient
            # clip alpha to be all > 0
            alpha = np.where(alpha < 0, 0, alpha)
            # clip alpha to be < C
            alpha = np.where(alpha > self.C, self.C, alpha)

        # save support vectors and corresponding labels
        self.alpha, self.sv, self.label = alpha[alpha > 0], X[alpha > 0, :], y[alpha > 0]
        return alpha, self.sv, self.label

    def predict(self, X):
        pass


def rbf(xi, xj, sigma):
    return np.exp(-np.linalg.norm(xi - xj) ** 2 / (2 * sigma ** 2))


if __name__ == '__main__':


    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")
    y = df['quality']
    df.drop(columns=['quality'], inplace=True)
    y = np.where(y > 5, 1, -1)
    X = df.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    normalizer = preprocessing.Normalizer()
    normalized_train_X = normalizer.fit_transform(X_train)
    normalized_test_X = normalizer.transform(X_test)

    my_svm = svc(sigma=100, kernel='rbf', C=3, n_iterations=10000, lr=0.0001, momentum=0.001)
    my_svm.fit(normalized_train_X, y_train)
    print(my_svm.alpha)
=======
from utils import performance
from sklearn.model_selection import train_test_split
from svm import SVM


if __name__ == '__main__':
    spectf_train = pd.read_csv(r'../data/SPECTF_train.csv')
    spectf_test = pd.read_csv(r'../data/SPECTF_test.csv')



    svm = SVM()
    svm.fit()
>>>>>>> 3a7bdf48766437d9d3471eb8133cf14dd8c282ca
