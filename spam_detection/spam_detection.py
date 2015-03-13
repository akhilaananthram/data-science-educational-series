import argparse

import pandas as pd
import numpy as np
import nltk
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold

def parse_args():
    parser = argparse.ArgumentParser(description="Recommendation Engine Example")
    parser.add_argument("--path", dest="path", required=True, type=str,
        help="Path to file. REQUIRED")
    parser.add_argument("--lowercase", dest="lower", action="store_true",
        help="Toggle use of lowercase")
    parser.add_argument("--classifier", dest="classifier", default="multinomial",
        choices=["multinomial", "bernoulli", "gaussian", "svm", "logistic"],
        help="Type of classifier to use")

    return parser.parse_args()

def read_data(filename):
    df = pd.read_csv(filename,sep='\t',header=None)
    tags = list(df[0])
    instances = list(df[1])
    instances = [i.decode('utf8') for i in instances]

    return np.array(instances), np.array(tags)

def tokenize(instances, lowercase=False):
    if lowercase:
        tokens = [nltk.word_tokenize(i.lower()) for i in instances]
    else:
        tokens = [nltk.word_tokenize(i) for i in instances]

    return tokens

if __name__=="__main__":
    args = parse_args()

    print "Reading in Data..."
    instances, tags = read_data(args.path)
    tokens = tokenize(instances, args.lower)
    frequencies = np.array([FreqDist(t) for t in tokens])

    overall_accuracy = 0.0
    total = 0.0
    kf = KFold(len(frequencies), n_folds=4)
    for train, test in kf:
        train_x = frequencies[train]
        train_y = tags[train]
        test_x = frequencies[test]
        test_y = tags[test]

        # Generate vocab
        print "Generating Vocabulary..."
        vocab = set()
        for i in train_x:
            for w in i.keys():
                vocab.add(w)
        vocab = list(vocab)
        vocab_to_index = {w:i for i,w in enumerate(vocab)}

        print "Generating Count Matrix..."
        train_vector = np.zeros((len(train_x), len(vocab)))
        for i in xrange(len(train_x)):
            f = train_x[i]
            for w, c in f.iteritems():
                j = vocab_to_index[w]
                train_vector[i,j] = c

        test_vector = np.zeros((len(test_x), len(vocab)))
        for i in xrange(len(test_x)):
            f = test_x[i]
            for w, c in f.iteritems():
                # Do not count words that were not in the training set
                if w in vocab_to_index:
                    j = vocab_to_index[w]
                    test_vector[i,j] = c

        print "Training TFIDF..."
        tfidf = TfidfTransformer()
        train_tfidf = tfidf.fit_transform(train_vector)
        test_tfidf = tfidf.transform(test_vector)

        print "Training Classifier..."
        if args.classifier=="multinomial":
            classifier = MultinomialNB()
        elif args.classifier=="bernoulli":
            classifier = BernoulliNB()
        elif args.classifier=="gaussian":
            classifier = GaussianNB()
        elif args.classifier=="svm":
            classifier = SVC()
        elif args.classifier=="logistic":
            classifier = LogisticRegression()

        classifier.fit(train_tfidf.toarray(), train_y)

        print "Test Classifier..."
        accuracy = classifier.score(test_tfidf.toarray(), test_y)
        print accuracy
        overall_accuracy += accuracy * len(test_y)
        total += len(test_y)

    print "Average Accuracy: {}".format(overall_accuracy / total)
