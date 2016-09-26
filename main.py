import logging
import cPickle as pkl
import msgpack
import os

from nose.tools import assert_equal, assert_true
import numpy as np
from numpy.testing import assert_allclose

import evaluate
import glove

from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s\t%(message)s")

# load HQMS medical concepts corpus
def load_corpus(data_path):
    if os.path.isfile('./data/corpus.pkl'):
        test_corpus = pkl.load(open('./data/corpus.pkl', 'r'))
        return test_corpus
    data = pkl.load(open(data_path, 'r'))
    concepts_data = data[['P321', 'P324', 'P327', 'P3291', 'P3294', 'P3297', 'P3281', 'P3287', 'P3271', 'P3274',
                          'P490', 'P4911', 'P4922', 'P4533', 'P4544', 'P45002', 'P45014', 'P45026', 'P45038', 'P45050']]
    concepts_data = concepts_data.fillna('')
    concepts_data = concepts_data.values
    concepts_data = [[str(code) for code in item if code != ''] for item in concepts_data]
    test_corpus = [' '.join(item) for item in concepts_data]
    pkl.dump(test_corpus, open('./data/corpus.pkl', 'w'))
    return test_corpus

data_path = '../medicalcare/data/dataclean.df'
test_corpus = load_corpus(data_path)

if os.path.isfile('./data/vocab'):
    vocab = msgpack.load(open('./data/vocab', 'r'))
else:
    vocab = glove.build_vocab(test_corpus)
    msgpack.dump(vocab, open('./data/vocab', 'w'))

if os.path.isfile('./data/cooccur'):
    cooccur = msgpack.load(open('./data/cooccur', 'r'))
else:
    cooccur = glove.build_cooccur(vocab, test_corpus, window_size=100)
    msgpack.dump(cooccur, open('./data/cooccur', 'w'))

if os.path.isfile('./data/id2word'):
    id2word = msgpack.load(open('./data/id2word', 'r'))
else:
    id2word = evaluate.make_id2word(vocab)
    msgpack.dump(id2word, open('./data/id2word', 'w'))

if os.path.isfile('./data/vectors.pkl'):
    W = pkl.load(open('./data/vectors.pkl', 'r'))
else:
    W = glove.train_glove(vocab, cooccur, vector_size=500, iterations=300)
    # Merge and normalize word vectors
    W = evaluate.merge_main_context(W)
    pkl.dump(W, open('./data/vectors.pkl', 'w'))

def predict():
    data = pkl.load(open(data_path, 'r'))
    demographics = np.zeros((len(data.index), 4))
    demographics = data[['P2', 'P5', 'P7', 'P27']].values

    assert len(test_corpus) == len(demographics)
    n_samples = len(test_corpus)
    mat = np.zeros((n_samples, W.shape[1]))
    for i in xrange(n_samples):
        for word in test_corpus[i].split(' '):
            mat[i] += W[vocab[word][0]]
    X = np.hstack((demographics, mat))
    y = data[['P782']].values
    X = np.asarray(X, dtype='float32')
    y = np.asarray(y, dtype='float32')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

    print 'Random Forest Regressiong'
    rf = ensemble.RandomForestRegressor(n_estimators=10, n_jobs=30)
    rf.fit(X_train, y_train.ravel())
    pkl.dump(rf, open('rfmodel.pkl', 'w'))
    y_rf = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test.squeeze(), y_rf))
    r2 = r2_score(y_test.squeeze(), y_rf)
    print rmse, r2

def test_similarity():
    similar = evaluate.most_similar(W, vocab, id2word, 'A16.504')
    logging.debug(similar)
    print similar[:10]

    # assert_equal('trees', similar[0])

if __name__ == '__main__':
    test_similarity()
