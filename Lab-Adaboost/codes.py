import numpy as np
import pandas as pd

class WeakClassifier:
    def __init__(self):
        self.tree = None 
        self.alpha = None
    
    def best_split(self, X, y, sample_weight):
        ''' 
        Find the optimal feature and threshold for splitting the data using Gini impurity.

        Args:
            X (pd.DataFrame): Data features, shape (n_samples, n_features).
            y (pd.Series): Data labels, shape (n_samples,).
            sample_weight (np.ndarray): Sample weights, shape (n_samples,).

        Returns:
            best_feature (str): Name of the feature selected for the split.
            best_threshold (float or int): Optimal threshold for the split.
            best_splits (tuple): Tuple containing masks for left and right splits (np.ndarray, np.ndarray).
        '''
        # TODO: Implement the function to find the best feature, threshold and splits to split the data based on the Gini impurity.
        def gini(y, sample_weight):
            unique_labels = y.unique()
            total_weight = np.sum(sample_weight)
            gini_impurity = 1.0
            for label in unique_labels:
                weight = np.sum(sample_weight[y == label])
                if total_weight > 0:
                    prob = weight / total_weight
                    gini_impurity -= prob ** 2
            return gini_impurity

        best_feature, best_threshold, best_splits = None, None, None
        min_gini = float('inf')

        for feature in X.columns:
            thresholds = np.unique(X[feature].values)
            for threshold in thresholds:
                left_mask = X[feature] <= threshold
                right_mask = X[feature] > threshold

                left_gini = gini(y[left_mask], sample_weight[left_mask])
                right_gini = gini(y[right_mask], sample_weight[right_mask])

                weighted_gini = (np.sum(sample_weight[left_mask]) * left_gini + np.sum(sample_weight[right_mask]) * right_gini) / np.sum(sample_weight)

                if weighted_gini < min_gini:
                    min_gini = weighted_gini
                    best_feature, best_threshold, best_splits = feature, threshold, (left_mask, right_mask)

        return best_feature, best_threshold, best_splits
        
        
    def fit(self, X, y, sample_weight):
        '''  
        Train the weak classifier on the data.

        Args:
            X (pd.DataFrame): Data features, shape (n_samples, n_features).
            y (pd.Series): Data labels, shape (n_samples,).
            sample_weight (np.ndarray): Sample weights, shape (n_samples,).

        Returns:
            None: Updates self.tree with the trained decision tree structure.
        '''
        best_feature, best_threshold, best_splits = self.best_split(X, y, sample_weight)
        # TODO: Create the tree as a nested dictionary
        self.tree = {
            'feature': best_feature,
            'threshold': best_threshold,
            'left_value': y[best_splits[0]].mode()[0],
            'right_value': y[best_splits[1]].mode()[0]
        }


    def predict(self, x):
        '''  
        Predict labels for the given data.

        Args:
            x (pd.DataFrame): Data features for prediction, shape (n_samples, n_features).

        Returns:
            predict_labels (np.ndarray): Predicted labels, shape (n_samples,).
        '''

        # Initialize list to store prediction results
        predict_labels = []

        # Predict label for each sample
        for i in range(len(x)):
            sample = x.iloc[i,:]

            # TODO: Predict the label of the sample
            predicted_label = self.tree['left_value'] if sample[self.tree['feature']] <= self.tree['threshold'] else self.tree['right_value']
            predict_labels.append(predicted_label)
        return np.array(predict_labels)
    
    
class Adaboost:
    def __init__(self, n_estimators=10):
        '''
        Initialize the Adaboost classifier.

        Args:
            n_estimators (int): Number of weak classifiers in the ensemble.
        '''
        # Number of weak classifiers to use
        self.n_estimators = n_estimators
        # List to store each weak classifier
        self.clfs = []
    
    # AdaBoost training process
    def fit(self, X, y):
        '''  
        Train the Adaboost ensemble.

        Args:
            X (pd.DataFrame): Data features, shape (n_samples, n_features).
            y (pd.Series): Data labels, shape (n_samples,).

        Returns:
            None: Updates the list of weak classifiers (self.clfs).
        '''
        n_samples, m_features = X.shape
    
        # Initialize weights for each sample
        w = np.ones(n_samples) / n_samples

        # Train each weak classifier
        for _ in range(self.n_estimators):
            clf = WeakClassifier()

            # 1. Fit the weak classifier using the current weights
            clf.fit(X, y, w)

            # TODO: 2. Predict the labels of the data using the weak classifier
            predictions = clf.predict(X)
            # TODO: 3. Calculate the error rate by comparing predictions to actual labels
            error = np.sum(w * (predictions != y)) / np.sum(w)

            # TODO: 4. Calculate alpha (the weight of the classifier in the final model)
            alpha = 0.5 * np.log((1 - error) / (error))  
            # TODO: 5. Update the sample weights to emphasize misclassified samples
            w *= np.exp(-alpha * y * predictions)
            # Normalize weights so that they sum to one
            w /= np.sum(w)

            # Store the classifier and its weight
            clf.alpha = alpha
            self.clfs.append(clf)
            

    def predict(self, X):
        '''  
        Predict the label for each sample using the Adaboost ensemble.

        Args:
            X (pd.DataFrame): Data features for prediction, shape (n_samples, n_features).

        Returns:
            predict_labels (np.ndarray): Predicted labels, shape (n_samples,).
        '''
        # TODO: 1. Compute the predictions of each weak classifier
        predicted_labels = 0
        for clf in self.clfs:
            clf_predictions = clf.predict(X)
        # TODO: 2. Compute the weighted sum of predictions from all classifiers
            predicted_labels += (clf.alpha * clf_predictions)
        # TODO: 3. Determine the final label based on the sign of the weighted sum (if x>0 return 1, else return -1)
        predicted_labels = np.where(predicted_labels > 0, 1, -1)
        return predicted_labels 