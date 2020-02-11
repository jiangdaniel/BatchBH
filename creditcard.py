#!/usr/bin/env python3

import multiprocessing as mp
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def main(args):
    seeds = range(args.n_splits)

    dfs = []
    null_pvals, nonnull_pvals = [], []
    with mp.Pool(args.n_cpus) as p:
        for df in tqdm(p.imap_unordered(create_df, seeds), total=len(seeds), ncols=75):
            null_pvals.append(df['pvals'][df['hypos'] == 0].to_numpy())
            nonnull_pvals.append(df['pvals'][df['hypos'] == 1].to_numpy())
            dfs.append(df)

    fig, axs = plt.subplots(ncols=2)
    pval_hist(axs[0], np.hstack(null_pvals), title='distribution of null $p$-values', ylabel='percentage of nulls')
    pval_hist(axs[1], np.hstack(nonnull_pvals), title='distribution of non-null $p$-values', ylabel='percentage of non-nulls')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(r'Empirical distribution of $p$-values in credit card dataset')
    fig.savefig('pval_distribution.png')

    df = pd.concat(dfs, keys=[f'df{i}' for i in seeds])
    df.to_pickle('pvals.pkl')


def create_df(seed):
    np.random.seed(seed)

    X_train1, X_train2, X_test, y_train1, y_train2, y_test = split(X, y, 0.6, 0.2)

    scaler = StandardScaler().fit(X_train1)
    X_train1, X_train2, X_test = map(scaler.transform, [X_train1, X_train2, X_test])

    clf = LogisticRegression(solver='lbfgs').fit(X_train1, y_train1)
    y_hat_train2_null = clf.predict_proba(X_train2[np.where(y_train2 == 0)[0]])[:, 1]
    y_hat_test = clf.predict_proba(X_test)[:, 1]

    pvals = np.zeros(y_hat_test.size)
    for i, p in enumerate(y_hat_test):
        pvals[i] = (p < y_hat_train2_null).mean()
    df = pd.DataFrame(np.vstack((pvals, y_test)).T, columns=('pvals', 'hypos'))
    return df


def split(X, y, train_size, test_size):
    '''Split X, y into train, val, and test datasets.'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=train_size/(1-test_size))
    return X_train, X_val, X_test, y_train, y_val, y_test


def pval_hist(ax, pvals, title, ylabel):
    ax.hist(pvals, bins=np.arange(0, 1.01, 0.1), weights=np.ones_like(pvals) / pvals.size)
    ax.set_xlabel(r'$p$-value')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_title(title)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='creditcard.csv',
        help='Path to credit card data csv file.')
    parser.add_argument('--n-splits', type=int, default=100,
        help='Number of random splits to generate p-values for. Default is 100.')
    parser.add_argument('--n-cpus', type=int, default=mp.cpu_count(),
        help='Number of cpus to use. Default is all available cpus.')
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    X = df[[f'V{i}' for i in range(1, 29)] + ['Amount']].to_numpy()
    y = df[['Class']].to_numpy().squeeze()

    main(args)
