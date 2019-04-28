
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
import numpy as np
from numpy import random
import pandas as pd
from sklearn.metrics import confusion_matrix

def c(psi):
    """
    Compute the average path length
    """
    if psi > 2:
        H = np.log(psi-1) + 0.5772156649
        return 2 * H - (2 * (psi-1.0)) / (psi * 1.0)
    elif psi == 2: return 1
    else: return 0

def path_length_ind(x, node, current_length=0):
    """
    Computes the path length for an individual node and and individual observation
    """
    if node.attribute is None:
        return current_length + c(node.size)*1.0

    if x[node.attribute] < node.split:
        return path_length_ind(x, node.left, current_length=current_length+1)
    else:
        return path_length_ind(x, node.right, current_length=current_length+1)

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        height_limit = np.ceil(np.log2(self.sample_size))

        trees = []
        for i in range(self.n_trees):
            sample = X[random.choice(len(X), self.sample_size, replace=False), :]
            tree = IsolationTree(height_limit=height_limit)
            tree.fit(sample, improved=improved)
            trees.append(tree)

        self.trees = trees
        return self

    def path_length(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        path_length_list = []
        for x in X:
            path_lengths = []
            for tree in self.trees:
                path_lengths.append(path_length_ind(x, tree.root))
            path_length_list.append(np.mean(path_lengths))
            
        return np.array(path_length_list).reshape(-1, 1)

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        path_length = self.path_length(X)
        return np.power(2, (-path_length)/c(self.sample_size))

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        scores[scores >= threshold] = 1
        scores[scores < threshold] = 0
        return scores

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        scores = self.anomaly_score(X)
        return self.predict_from_anomaly_scores(scores, threshold)

class IsolationNode:
    """
    A single node in an isolation forest
    """
    def __init__(self, data, attribute=None, split=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
        self.attribute = attribute
        self.split = split
        self.size = len(data)


class IsolationTree:
    def __init__(self, height_limit, n_nodes=1):
        self.height_limit = height_limit
        self.n_nodes = n_nodes

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        self.root = self.build_tree(data=X, improved=improved)
        return self.root

    def build_tree(self, data, improved=False, height=0):
        if data.shape[0] <= 1 or height==self.height_limit:
            return IsolationNode(attribute=None, data=data, split=None, left=None, right=None)

        else:
            self.n_nodes += 2
            
            attribute = random.randint(data.shape[1])
            data_attribute = data[:, attribute]
            attribute_max = data_attribute.max()
            attribute_min = data_attribute.min()
            
            if improved:
                splits = [random.uniform(attribute_min, attribute_max) for i in range(5)]
                data_size = data.shape[0]
                left_data_size = [data[data_attribute < split].shape[0] for split in splits]
                right_data_size = [data[data_attribute >= split].shape[0] for split in splits]
                size_diff = [np.abs(left - right) for left, right in zip(left_data_size, right_data_size)]
                split = splits[np.argmax(size_diff)]

            else:
                split = random.uniform(attribute_min, attribute_max)

            tree_left = self.build_tree(data[data_attribute < split], improved=improved, height=height+1)
            tree_right = self.build_tree(data[data_attribute >= split], improved=improved, height=height+1)
            
            return IsolationNode(attribute=attribute, data=data, split=split, left=tree_left, right=tree_right)



def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    TPR = 0.0
    threshold = 1.0

    while TPR < desired_TPR:
        temp = scores.copy()
        predict = np.where(scores >= threshold, 1.0, 0.0)
        confusion = confusion_matrix(y, predict)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / ((TP + FN) * 1.0)
        FPR = FP / ((FP + TN) * 1.0)
        if TPR < desired_TPR:
            threshold = threshold - 0.005

    return threshold, FPR

