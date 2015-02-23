import random
import argparse
import math

import pandas as pd # C libraries for efficient tabular data
import numpy as np
import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import *
from sklearn import cross_validation

def parse_args():
    parser = argparse.ArgumentParser(description="Recommendation Engine Example")
    parser.add_argument("--path", dest="path", required=True, type=str,
        help="Path to file. REQUIRED")
    parser.add_argument("--max-hidden", dest="max_hidden", required=True, type=int,
        help="Maximum number of attributes to hide. REQUIRED")
    parser.add_argument("--k", dest="k", default=3,
        help="Number of neighbors to consider")
    parser.add_argument("--distance", dest="distance", default='cosine',
        choices=['jaccard','euclidean', 'cosine'], help="Distance metric")
    parser.add_argument("--weight", dest="weight", default='none',
        choices=['none', 'harmonic', 'exponential', 'gaussian'],
        help="Weight function for neighbors")
    parser.add_argument('--visualize', action='store_true', dest="visualize",
        help="Show visualization")

    return parser.parse_args()

def read_data(filename):
    print "Reading in {} ...".format(filename)
    
    # dataTable is a pandas data frame containing our data
    dataTable = pd.read_csv(filename)

    num_instances = len(dataTable['Timestamp'])

    # Delete the timestamp as it adds no value
    del dataTable['Timestamp']

    names = list(dataTable.keys())
    num_attributes = len(names)
    instances = np.zeros((num_instances, num_attributes))

    # Generate features
    for i, k in enumerate(names):
        attribute = dataTable[k]
        for j, val in enumerate(attribute):
            instances[j][i] = val

    return instances, names

def correlation_check(instances, names):
    # Take transpose of instances
    attributes = np.transpose(instances)

    N = len(attributes)
    fig = tls.make_subplots(rows=N, cols=N, print_grid=False,
                                  shared_xaxes=True, shared_yaxes=True)

    for i in xrange(N):
        # Scatter plot
        for j in xrange(i):
            scatt = Scatter(
                x=attributes[i],
                y=attributes[j],
                mode='markers'
            )
            fig.append_trace(scatt, row=i+1, col=j+1)

        # Histogram for i
        hist = Histogram(
            x=attributes[i]
        )
        fig.append_trace(hist, row=i+1, col=i+1)

        # Correlation graph
        for j in xrange(i + 1, N):
            # corrcoef returns the normalized covariance matrix
            corr = np.corrcoef(attributes[i], attributes[j])[0][1]
            trace = Scatter(
                x = [0],
                y = [0],
                mode='text',
                text = [str(corr)],
                textposition='top'
            )
            fig.append_trace(trace, row=i+1, col=j+1)

    fig['layout']['showlegend'] = False
    plot_url = py.plot(fig, filename='correlation-check')

# Sampling
def reservoir_sampling(instance, n):
    size = min(len(instance), n)
    indices = [-1] * n

    # Select n values from the instance to keep
    for i in xrange(n):
        indices[i] = i

    for i in xrange(n, len(instance)):
        j = random.randrange(0, i)

        if j < n:
            indices[j] = i

    # Set all other positions to 0
    hidden = np.zeros(len(instance))
    for i in indices:
        hidden[i] = instance[i]

    return hidden

def hide_attributes(instances, max_hidden):
    new_instances = []

    for inst in instances:
        n = random.randrange(1, max_hidden)
        new_instances.append(reservoir_sampling(inst, n))

    return np.array(new_instances)

# Distance Functions
def jaccard(A, B):
    numerator = 0.0
    denominator = 0.0
    for a, b in zip(A, B):
        # 0 stands for missing data
        if a!=0 and b!=0:
            numerator += min(a,b)
            denominator += max(a,b)

    # Subtracting from 1 so 0 = identical and 1 = opposite
    return 1 - (numerator / denominator)

def euclidean(A, B):
    total = 0.0
    for a, b in zip(A, B):
        # 0 stands for missing data
        if a!=0 and b!=0:
            total += (a - b) ** 2

    # 0 = identical
    return math.sqrt(total)

def cosine_similarity(A, B):
    # Subtracting from 1 so 0 = identical and 1 = opposite
    return 1 - (np.dot(A,B) / (np.linalg.norm(A) * np.linalg.norm(B)))

# Weight Functions
def no_weight(i):
    return 1.0

def harmonic(i):
    return 1.0 / (i + 1)

def exponential_decay(i):
    return math.exp(-i)

def gaussian(i, mu = 0.0, sig=3):
    return math.exp(-((i - mu) ** 2) / (2 * (sig ** 2)))

# Classifier
def predict(instance, train, distance, weight, k):
    # Sort the training instances
    neighbors = sorted(train, key= lambda x:distance(instance,x))
    nearest = neighbors[:k]

    predictions = []
    for i, val in enumerate(instance):
        # Missing entry
        if val == 0:
            # Additional bucket for 0.  It will remain empty
            votes = np.zeros(8)
            weights = []
            for j, n in enumerate(nearest):
                # Entry is not missing
                if n[i] != 0:
                    # Farther neighbors affect it less
                    w = weight(j)
                    weights.append(w)
                    label = n[i]
                    votes[label] += w

            # Neighbors may not have anything for this either
            if len(weights)!=0:
                p = np.argmax(votes)
                predictions.append((i,p))
        
    return predictions

# Evaluation
def calculate_accuracy(predictions, actual):
    exact = 0.0
    error = 0.0
    total = 0.0
    for i, p in predictions:
        if p!=0:
            total += 1
            e = abs(p - actual[i])
            # exact match
            if e == 0:
                exact += 1

            # if e == 6, exact opposite
            error += (1.0 / 6) * e

    weighted_accuracy = (total - error) / total
    accuracy = exact / total

    return accuracy, weighted_accuracy, total

if __name__=="__main__":
    args = parse_args()

    if args.distance == "jaccard":
        distance = jaccard
    elif args.distance == "euclidean":
        distance = euclidean
    elif args.distance == "cosine":
        distance = cosine_similarity

    if args.weight == "none":
        weight = no_weight
    elif args.weight == "harmonic":
        weight = harmonic
    elif args.weight == "exponential":
        weight = exponential_decay
    elif args.weight == "gaussian":
        weight = gaussian

    # Read data and create features
    instances, names = read_data(args.path)

    if args.visualize:
        correlation_check(instances, names)

    hidden = hide_attributes(instances, args.max_hidden)

    kf = cross_validation.KFold(len(hidden), n_folds=4)
    totalAcc = []
    totalWeightedAcc = []
    for trainidx, testidx in kf:
        train, test = hidden[trainidx], hidden[testidx]
        actual = instances[testidx]
        predicted = 0.0
        total_accuracy = 0.0
        total_weighted_accuracy = 0.0
        for t, a in zip(test, actual):
            predictions = predict(t, train, distance, weight, args.k)
            accuracy, weighted_accuracy, total = calculate_accuracy(predictions, a)
            predicted += total
            total_accuracy += accuracy * total
            total_weighted_accuracy += weighted_accuracy * total

        total_accuracy /= predicted
        total_weighted_accuracy /= predicted
        totalAcc.append(total_accuracy)
        totalWeightedAcc.append(total_weighted_accuracy)
        print "Accuracy: {}, Weighted Accuracy: {}".format(total_accuracy, total_weighted_accuracy)

    print "Aver Acc: {}, Weighted Aver Acc: {}".format(sum(totalAcc)/len(totalAcc), sum(totalWeightedAcc)/len(totalWeightedAcc))
