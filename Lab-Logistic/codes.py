import numpy as np
from typing import Tuple
from vis_util import plot_progress


def sigmoid(x):
    ''' 
    Sigmoid function.
    '''
    return 1 / (1 + np.exp(-x))

class LogisticRegression():
    ''' 
    Logistic Regression
    '''
    def __init__(
        self,
        plot: bool = False
    ) -> None:
        
        self.w = None # random intialize w
        self.lr = None # learning rate
        self.reg = None # regularization parameter
        self.plot = plot

    def predict(
        self,
        x: np.array
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        ''' 
        Logistic Regression (LR) prediction.
        
        Arguments:
            x : (n, d + 1), where n represents the number of samples, d the number of features

        Return:
            prob: (n,), LR probabilities, where prob[i] is the probability P(y=1|x,w) for x[i], from [0, 1]
            pred: (n,), LR predictions, where pred[i] is the prediction for x[i], from {-1, 1}

        '''
        # implement predict method,
        # !! Assume that : self.w is already given.
        
        # TODO: first, you should compute the probability by invoking sigmoid function
        prob = sigmoid(x @ self.w)

        # TODO: second, you should compute the prediction (W^T * x >= 0 --> y = 1, else y = -1)
        pred = np.where(prob >= 0.5, 1, -1)
        return prob, pred

    def fit(
        self,
        x: np.array,
        y: np.array,
        n_iter: int,
        lr: float,
        reg: float,
    ) -> None:
        ''' 
        Logistic Regression (LR) training.

        Arguments:
            x : (n, d), where n represents the number of training samples, d the number of features
            y : (n,), where n represents the number of samples
            n_iter : number of iteration
            lr : learning rate
            reg : regularization parameter
            
        Return:
            None
        '''
        self.lr = lr
        self.reg = reg

        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1) # add bias term
        self.w = np.random.normal(0, 1, x.shape[1]) # random intialize w
        loss_history = []
        w_module_history = []

        
        for i in range(n_iter):
            
            # update the weight 
            self.update(x, y)
            

            # plot loss and w module every 10 iterations
            if i % 10 == 0:
                # compute the loss
                loss = self.calLossReg(x, y)
                
                loss_history.append(loss)

                w_module_history.append(np.linalg.norm(self.w))
                if self.plot:
                    print("iter: {}, loss: {}, w_module: {}".format(i, loss, w_module_history[-1]))
                    plot_progress(i, loss_history,w_module_history, x, y, self.w)
    

    def update(
        self,
        x: np.array,
        y: np.array,
    ) -> None:
        
        '''
        Update the parameters--weight w
        Arguments:
            x: (n, d+1), training samples, where n represents the number of training samples, d the number of features
            y: (n,), training labels, where n represents the number of training samples

        Return:
            None
        '''

        # implement gradient descent algorithm

        # TODO: 1. compute the gradient
        prob = sigmoid(x @ self.w)
        gradient = (x.T @ (prob - (y + 1) / 2)) / x.shape[0] + self.reg * self.w
        # TODO: 2. update the weight 
        self.w -= self.lr * gradient


    def calLossReg(
        self,
        x: np.array,
        y: np.array,
    ):
        ''' 
        Compute the loss

        Arguments:
            x: (n, d+1), training samples, where n represents the number of training samples, d the number of features
            y: (n,), training labels, where n represents the number of training samples

        Return:
            loss: float, the loss value
        '''
        # TODO: compute the Logistic Regression loss, including regularization term
        # !! Note that the label y is from {-1, 1}
        prob = sigmoid(x @ self.w)
        y = (y + 1) / 2
        log_likelihood = -np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob)) +  self.reg * np.linalg.norm(self.w) ** 2
        return log_likelihood


    def calLoss(
        self,
        x: np.array,
        y: np.array,
    ):
        ''' 
        Compute the loss

        Arguments:
            x: (n, d+1), training samples, where n represents the number of training samples, d the number of features
            y: (n,), training labels, where n represents the number of training samples

        Return:
            loss: float, the loss value
        '''
        # TODO: compute the Logistic Regression loss, including regularization term
        # !! Note that the label y is from {-1, 1}
        prob = sigmoid(x @ self.w)
        y = (y + 1) / 2
        log_likelihood = -np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob)) # +  self.reg * np.linalg.norm(self.w) ** 2
        return log_likelihood

def compute_accuracy(
    y_pred: np.array,
    y_true: np.array,
) -> float:
    '''
    Compute the accuracy

    Arguments:
        y_pred: (n,), where n represents the number of samples
        y_true: (n,), where n represents the number of samples
    Return:
        acc: float, the accuracy
    '''
    # TODO: compute the accuracy
    acc = np.mean(y_pred == y_true)

    return acc
 