#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
from pathlib import Path
import pickle
import time
from itertools import repeat
import multiprocessing as mp
import argparse
from collections import namedtuple
import random

import numpy as np
import pandas as pd
from numpy.random import randn
from scipy.stats import norm
from scipy.linalg import toeplitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('base.mplstyle')
from tqdm import tqdm

from utils import bh, storey_bh, timestamp, gamma, transpose, two_lvl_defaultdict, save

np.random.seed(0)
random.seed(0)


def prds(seq):
    cov = (np.eye(args.prds_size) + np.ones((args.prds_size, args.prds_size))) / 2
    Z = np.zeros(seq.size)
    for i, sub_seq in enumerate(np.split(seq, seq.size//args.prds_size)):
        mu = 3 * (sub_seq == 1).astype(float)
        Z[i*args.prds_size:(i+1)*args.prds_size] = np.random.multivariate_normal(mu, cov)
    return Z


Z_FUNCS = {
    'Mean2': lambda seq: randn(seq.size) + 2 * seq,  # alternative mean is 2
    'Mean3': lambda seq: randn(seq.size) + 3 * seq,  # alternative mean is 3
    'GaussianMean': lambda seq: randn(seq.size) + randn(seq.size) * np.sqrt(2 * np.log(seq.size) * seq),  # alternative mean is gaussian
    'Constant': lambda seq: randn(seq.size) * (1-seq) + np.log(seq.size) * seq,  # alternative is constant
    'PRDS': prds,
}

P_FUNCS = {
    '1S': lambda z: norm.sf(z),  # one sided p-values
    '2S': lambda z: 2 * norm.sf(np.abs(z)),  # two sided p-values
}

def get_sequences(pi1, args):
    if args.real:
        hypo_sequences = []
        pval_sequences = []

        df = pd.read_pickle(args.real)
        for i in range(args.real_splits):
            key = f'df{i}'
            hypo_sequences.append(df.xs(key)['hypos'].to_numpy())
            pval_sequences.append(df.xs(key)['pvals'].to_numpy())
    else:
        hypo_sequences = [np.random.binomial(1, pi1, args.n_hypos) for _ in range(args.n_trials)]

        p_func = P_FUNCS[args.task.split('-')[0]]
        z_func = Z_FUNCS[args.task.split('-')[1]]
        pval_sequences = []
        pval_sequences = [p_func(z_func(seq)) for seq in hypo_sequences]
    return hypo_sequences, pval_sequences

LINESTYLES = iter(['-', '--', '-.', ':'])

def setup_pwr_plot(ax, xlabel, xticks=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$\\mathrm{Power}$')
    ax.set_xlim(0)
    ax.set_ylim(0, 1)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_yticks(np.linspace(0, 1, 6))

def setup_fdr_plot(ax, xlabel, alpha, unbounded=False, xticks=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$\\mathrm{FDR}$')
    ax.set_xlim(0)
    if xticks is not None:
        ax.set_xticks(xticks)
    if not unbounded:
        ax.set_ylim(0, 2 * alpha)
        ax.set_yticks(np.linspace(0, 2*alpha, 6))
    ax.axhline(y=alpha, color='black')  # add a horizontal black line at alpha

def plot_mean_std(ax, xs, ys, ls, label):
    '''Utility function for plotting mean and standard deviation. The mean and
    standard deviation is calculated across the 0-axis of YS.'''
    assert len(xs) == ys.shape[1]
    ys_mean, ys_std = ys.mean(0), ys.std(0)
    line = ax.plot(xs, ys_mean, label=label, ls=ls)[0]
    ax.fill_between(xs, ys_mean - ys_std, ys_mean + ys_std, alpha=0.1, color=line.get_c())


def main(args):
    # online algorithms (e.g., Lord++, Saffron) must have a batch size of 1
    # tuples of the form (class, printed name, latex name, matplotlib name, batch sizes)
    if args.testers == 'nonadaptive':
        tester_tuples = [
            (BatchBH, 'Batch BH', '$\\batchbh$', '$\\mathrm{Batch}_{\\mathrm{BH}}$', [10, 100, 1000]),
            (Lord, 'LORD', 'LORD', '$\\mathrm{LORD}$', [1]),
            ]
    elif args.testers == 'adaptive':
        tester_tuples = [
            (BatchStBH, 'Batch St-BH', '$\\batchsbh$', '$\\mathrm{Batch}_{\\mathrm{St-BH}}$', [10, 100, 1000]),
            (Saffron, 'SAFFRON', 'SAFFRON', '$\\mathrm{SAFFRON}$', [1]),
            ]
    elif args.testers == 'batchbh':
        tester_tuples = [
            (BatchBH, 'Batch BH', '$\\batchbh$', '$\\mathrm{Batch}_{\\mathrm{BH}}$', [10, 100, 1000]),
            ]
    elif args.testers == 'batchsbh':
        tester_tuples = [
            (BatchStBH, 'Batch St-BH', '$\\batchsbh$', '$\\mathrm{Batch}_{\\mathrm{St-BH}}$', [10, 100, 1000]),
            ]
    elif args.testers == 'bh':
        tester_tuples = [
            (BH, 'BH', 'BH', '$\\mathrm{BH}$', [10, 100, 1000]),
            ]
    elif args.testers == 'sbh':
        tester_tuples = [
            (StoreyBH, 'Storey-BH', 'Storey-BH', '$\\mathrm{Storey}$-$\\mathrm{BH}$', [10, 100, 1000]),
            ]
    elif args.testers == 'prds-100':
        tester_tuples = [
            (BatchPrds, 'Batch BH PRDS', '\\batchbh^{\\text{PRDS}}', '$\\mathrm{Batch}_{\\mathrm{BH}}^{\\mathrm{PRDS}}$', [100]),
            (Lond, 'LOND', 'LOND', '$\\mathrm{LOND}$', [1]),
        ]
    elif args.testers == 'prds-1000':
        tester_tuples = [
            (BatchPrds, 'Batch BH PRDS', '\\batchbh^{\\text{PRDS}}', '$\\mathrm{Batch}_{\\mathrm{BH}}^{\\mathrm{PRDS}}$', [1000]),
            (Lond, 'LOND', 'LOND', '$\\mathrm{LOND}$', [1]),
        ]
    else:
        raise Exception

    testers = []
    Tester = namedtuple('Tester', ['cls', 'size', 'label', 'tex_label', 'mpl_label', 'ls', 'pwrs', 'fdrs', 'monotones'])
    for cls, name, tex_name, mpl_name, batch_sizes in tester_tuples:
        for size, ls in zip(batch_sizes, LINESTYLES):
            label = name + f' (10^{int(np.log10(size))}-size)' if issubclass(cls, BatchTester) else name
            tex_label = tex_name + f' ($10^{int(np.log10(size))}$-size)' if issubclass(cls, BatchTester) else tex_name
            mpl_label = mpl_name + f' $(10^{int(np.log10(size))}- ' + '\\mathrm{size})$' if issubclass(cls, BatchTester) else mpl_name
            testers.append(Tester(cls, size, label, tex_label, mpl_label, ls, [], [], []))

    data = two_lvl_defaultdict()
    for pi1 in args.pi1s:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        hypo_sequences, pval_sequences = get_sequences(pi1, args)
        for tester in testers:
            with mp.Pool(args.n_cpus) as pool:
                start = time.time()
                fdrs_hist, fdhs_hist, pwrs_hist, alps_hist, rejs_hist, R_diffs_hist = transpose(
                    pool.starmap(run_trial, zip(repeat(tester.cls), hypo_sequences, pval_sequences, repeat(tester.size), repeat(args.alpha)))
                    )
                print(f'Finished pi1={pi1}, {tester.label} in {round(time.time()-start)} seconds.')
                if args.monotone:
                    rand_hypos = [random.choice(range(args.n_hypos)) for _ in range(args.n_trials)]
                    monotone_count = sum(
                        pool.starmap(is_monotone, zip(repeat(tester.cls), hypo_sequences, pval_sequences, repeat(tester.size), repeat(args.alpha), rejs_hist, rand_hypos))
                        )
            if args.monotone and issubclass(tester.cls, BatchTester):
                tester.monotones.append(monotone_count / args.n_trials * 100)
                print(f'{monotone_count}/{args.n_trials} ({round(100*monotone_count/args.n_trials, 2)}%) monotone')
            data[tester.label][pi1]['fdr'] = fdrs_hist
            data[tester.label][pi1]['pwr'] = pwrs_hist

            xs = np.arange(tester.size, args.n_hypos+1, tester.size)
            if xs[-1] != args.n_hypos:
                xs = np.append(xs, np.array([args.n_hypos]))

            plot_mean_std(axs[0], xs, np.vstack(pwrs_hist), ls=tester.ls, label=tester.mpl_label)
            plot_mean_std(axs[1], xs, np.vstack(fdrs_hist), ls=tester.ls, label=tester.mpl_label)

            tester.pwrs.append([pwrs[-1] for pwrs in pwrs_hist])
            tester.fdrs.append([fdrs[-1] for fdrs in fdrs_hist])

            if args.rdiff:
                fig_rdiff, ax_rdiff = plt.subplots(nrows=1, ncols=1)
                ax_rdiff.scatter(np.arange(len(R_diffs_hist[0])), R_diffs_hist[0], label=tester.mpl_label, s=30)
                ax_rdiff.set_xlabel('$t$')
                ax_rdiff.set_ylabel('$R_t^+ - R_t$')
                if ax_rdiff.get_xticks()[1] - ax_rdiff.get_xticks()[0] < 1:
                    ax_rdiff.set_xticks(np.arange(len(R_diffs_hist[0])))
                save(fig_rdiff, Path(OUT_DIR, f'pi1_{int(pi1*100):02d}_batch_{tester.size}_rdiff'))
                plt.close(fig_rdiff)

        setup_pwr_plot(axs[0], xlabel='$t$')
        setup_fdr_plot(axs[1], xlabel='$t$', alpha=args.alpha, unbounded=args.testers in ('bh', 'sbh'))
        fig.legend(axs[0].lines, [x.get_label() for x in axs[0].lines], ncol=2)
        save(fig, Path(OUT_DIR, f'pi1_{int(pi1*100):02d}'))
        plt.close(fig)

    with Path(OUT_DIR, 'data.pkl').open('wb') as f:
        pickle.dump(data, f)

    if args.real:
        pwr_strs, fdr_strs = [], []
        for tester in testers:
            pwr_strs.append((f'{round(np.mean(tester.pwrs), 3):.3f} $\\pm$ {round(np.std(tester.pwrs), 3):.3f}'))
            fdr_strs.append((f'{round(np.mean(tester.fdrs), 3):.3f} $\\pm$ {round(np.std(tester.fdrs), 3):.3f}'))
        print(pd.DataFrame(
            columns=('Power', 'FDR'),
            index=[x.tex_label for x in testers],
            data=zip(pwr_strs, fdr_strs)
            ).to_latex(escape=False, column_format='@{}lll@{}'))

    if args.monotone:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_xlabel('$\\pi_1$')
        ax.set_ylabel('% of trials that are monotone', fontsize=19)
        ax.set_ylim(95, 100)
        ax.set_yticks(np.arange(95, 101))
        for tester in testers:
            if issubclass(tester.cls, BatchTester):
                ax.plot(args.pi1s, tester.monotones, ls=tester.ls, label=tester.mpl_label)
        fig.legend(ax.lines, [x.get_label() for x in ax.lines], ncol=2)
        save(fig, Path(OUT_DIR, 'monotone'))
        plt.close(fig)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    for tester in testers:
        plot_mean_std(axs[0], args.pi1s, np.vstack(tester.pwrs).T, ls=tester.ls, label=tester.mpl_label)
        plot_mean_std(axs[1], args.pi1s, np.vstack(tester.fdrs).T, ls=tester.ls, label=tester.mpl_label)

    xticks = np.arange(0, max(args.pi1s) + 1e-6, 0.1)
    setup_pwr_plot(axs[0], xlabel='$\\pi_1$', xticks=xticks)
    setup_fdr_plot(axs[1], xlabel='$\\pi_1$', alpha=args.alpha, unbounded=args.testers in ('bh', 'sbh'), xticks=xticks)

    fig.legend(axs[0].lines, [x.get_label() for x in axs[0].lines], ncol=2)
    save(fig, Path(OUT_DIR, 'pi1s'))
    plt.close(fig)


def run_trial(tester_cls, hypo_seq, pval_seq, batch_size, alpha):
    '''Simulates a single trial of applying TESTER_CLS to the HYPO_SEQ and PVAL_SEQ
    chunked by BATCH_SIZE at test level ALPHA. Returns a sequence (same size as HYPO_SEQ)
    of FDRs, FDP hats, powers, and test levels.'''

    splits = np.cumsum(batch_size * np.ones(hypo_seq.size + 1))
    splits = splits[:np.where(splits >= hypo_seq.size)[0][0]].astype(int)

    tester = tester_cls(alpha)
    tp, fp, fn = 0, 0, 0
    fdrs, fdhs, pwrs, alps, rejs, R_diffs = [], [], [], [], [], []
    for hypos, pvals in zip(np.split(hypo_seq, splits), np.split(pval_seq, splits)):
        preds, fdh, alp, R_diff = tester(pvals)
        tp += (hypos * preds).sum()  # reject false null
        fp += ((1-hypos) * preds).sum()  # reject true null
        fn += (hypos * (1-preds)).sum()  # accept false null
        fdrs.append(fp / max(tp + fp, 1))
        pwrs.append(tp / max(tp + fn, 1))
        fdhs.append(fdh)
        alps.append(alp)
        rejs.append(preds.sum())
        R_diffs.append(R_diff)
    return fdrs, fdhs, pwrs, alps, rejs, R_diffs


def is_monotone(tester_cls, hypo_seq, pval_seq, batch_size, alpha, rejs, rand_hypo):
    rand_p = pval_seq[rand_hypo]
    rand_batch = rand_hypo // batch_size
    if rand_batch >= len(rejs) - 1:
        return True
    pval_seq[rand_hypo] = 0
    _, _, _, _, new_rejs, _ = run_trial(tester_cls, hypo_seq, pval_seq, batch_size, alpha)
    pval_seq[rand_hypo] = rand_p
    return sum(rejs[rand_batch+1:]) <= sum(new_rejs[rand_batch+1:])


class SingleTester(object):
    def __init__(self):
        pass

    def test(self, pval):
        pass

    def __call__(self, pval):
        return self.test(pval)


class BatchTester(object):
    def __init__(self):
        pass

    def test(self, pvals):
        pass

    def __call__(self, pvals):
        return self.test(pvals)


#### NON-ADAPTIVE ####


class BH(BatchTester):
    '''Standard BH.'''
    def __init__(self, alpha):
        self.alpha = alpha

        # parameters for FDP hat calculation
        self.R_sums = np.zeros(2)
        self.R_total_sum = 0
        self.t = 1

    def test(self, pvals):
        num_rejects, p_threshold = bh(pvals, self.alpha)

        if self.R_sums.size == self.t:
            self.R_sums = np.append(self.R_sums, np.zeros(self.t))
        self.R_sums[self.t] = self.R_total_sum
        self.R_sums[1:self.t] += num_rejects
        self.R_total_sum += num_rejects

        self.t += 1
        fdh = pvals.size * self.alpha * (np.ones(self.t-1) / (pvals.size + self.R_sums[1:self.t])).sum()
        return (pvals <= p_threshold).astype(float), fdh, self.alpha, 0


class BatchBH(BatchTester):
    def __init__(self, alpha):
        self.alpha = alpha

        # parameters for test level calculation
        self.R_pluses = np.zeros(32)
        self.R_total_sum = 0
        self.R_sums = np.zeros(32)
        self.alphas = np.zeros(32)
        self.js = np.arange(32, dtype=float)
        self.t = 1

    def test(self, pvals):
        if self.R_pluses.size == self.t:
            # double array sizes
            self.R_pluses = np.append(self.R_pluses, np.zeros(self.t))
            self.R_sums = np.append(self.R_sums, np.zeros(self.t))
            self.alphas = np.append(self.alphas, np.zeros(self.t))
            self.js = np.arange(2 * self.t, dtype=float)

        if self.t == 1:
            alpha_t = self.alpha * self._gamma(self.t, pvals.size)
        else:
            alpha_t = self.alpha * self._gamma(self.js[1:self.t+1], pvals.size).sum() - (self.alphas[1:self.t] * self.R_pluses[1:self.t] / (self.R_pluses[1:self.t] + self.R_sums[1:self.t])).sum()
            alpha_t *= (pvals.size + self.R_total_sum) / pvals.size
        num_rejects, p_threshold = bh(pvals, alpha_t)

        self.R_sums[self.t] = self.R_total_sum
        self.R_sums[1:self.t] += num_rejects
        self.R_total_sum += num_rejects
        self.alphas[self.t] = alpha_t

        R_plus = 0
        for i, p in enumerate(pvals):
            pvals[i] = 0
            R_plus = max(R_plus, bh(pvals, alpha_t)[0])
            pvals[i] = p
        self.R_pluses[self.t] = R_plus

        self.t += 1
        fdh = (self.alphas[1:self.t] * self.R_pluses[1:self.t] / (self.R_pluses[1:self.t] + self.R_sums[1:self.t])).sum()  # fdp hat
        return (pvals <= p_threshold).astype(float), fdh, alpha_t, R_plus - num_rejects

    @staticmethod
    def _gamma(x, size):
        if size < 100:
            return gamma('poly', x)
        else:
            return gamma('half', x)


class Lord(SingleTester):
    def __init__(self, alpha):
        self.alpha, self.w0 = alpha, alpha / 2

        # parameters for test level calculation
        self.t = 1
        self.reject_times = np.zeros(128)
        self.rejects = 0

        # parameters for FDP hat calculation
        self.alpha_sum = 0

    def test(self, pval):
        alpha_t = self.w0 * self._gamma(self.t)
        if self.rejects > 0:
            alpha_t += (self.alpha - self.w0) * self._gamma(self.t - self.reject_times[1])
            if self.rejects > 1:
                alpha_t += self.alpha * self._gamma(self.t - self.reject_times[2:self.rejects+1]).sum()

        if pval[0] <= alpha_t:  # rejection
            prediction = np.ones(1)
            self.rejects += 1
            if self.rejects == self.reject_times.size:
                self.reject_times = np.append(self.reject_times, np.zeros_like(self.reject_times))
            self.reject_times[self.rejects] = self.t
        else:  # non-rejection
            prediction = np.zeros(1)

        self.t += 1
        self.alpha_sum += alpha_t
        fdh = self.alpha_sum / max(self.rejects, 1)  # fdp hat
        return prediction, fdh, alpha_t, 0

    @staticmethod
    def _gamma(x):
        return gamma('log', x)


#### ADAPTIVE ####


class StoreyBH(BatchTester):
    '''Standard StoreyBH.'''
    def __init__(self, alpha):
        self.alpha = alpha
        self.lmbda = 0.5

        # parameters for FDP hat calculations
        self.ks, self.R_sums = np.zeros(2), np.zeros(2)
        self.R_total_sum = 0
        self.t = 1

    def test(self, pvals):
        num_rejects, p_threshold = storey_bh(pvals, self.alpha, self.lmbda)

        if self.ks.size == self.t:
            self.ks = np.append(self.ks, np.zeros(self.t))
            self.R_sums = np.append(self.R_sums, np.zeros(self.t))
        self.ks[self.t] = (pvals > self.lmbda).sum() / (1 + (pvals > self.lmbda).sum() - (pvals.max() > self.lmbda).sum())
        self.R_sums[self.t] = self.R_total_sum
        self.R_sums[1:self.t] += num_rejects
        self.R_total_sum += num_rejects
        self.t += 1
        fdh = self.alpha * pvals.size * (self.ks[1:self.t] / (pvals.size + self.R_sums[1:self.t])).sum()
        return (pvals <= p_threshold).astype(float), fdh, self.alpha, 0


class BatchStBH(BatchTester):
    def __init__(self, alpha):
        self.alpha = alpha
        self.lmbda = 0.5

        # parameters for test level calculation
        self.ks = np.zeros(32)
        self.R_pluses = np.zeros(32)
        self.R_total_sum = 0
        self.R_sums = np.zeros(32)
        self.alphas = np.zeros(32)
        self.js = np.arange(32, dtype=float)
        self.t = 1

    def test(self, pvals):
        if self.ks.size == self.t:
            # double array sizes
            self.ks = np.append(self.ks, np.zeros(self.t))
            self.R_pluses = np.append(self.R_pluses, np.zeros(self.t))
            self.R_sums = np.append(self.R_sums, np.zeros(self.t))
            self.alphas = np.append(self.alphas, np.zeros(self.t))
            self.js = np.arange(2 * self.t, dtype=float)

        greater_sum = (pvals > self.lmbda).sum()
        self.ks[self.t] = greater_sum / (greater_sum + (pvals.max() <= self.lmbda).sum())

        if self.t == 1:
            alpha_t = self.alpha * self._gamma(self.t, pvals.size)
        else:
            alpha_t = self.alpha * self._gamma(self.js[1:self.t+1], pvals.size).sum() - (self.alphas[1:self.t] * self.ks[1:self.t] * self.R_pluses[1:self.t] / (self.R_pluses[1:self.t] + self.R_sums[1:self.t])).sum()
            alpha_t *= (pvals.size + self.R_total_sum) / pvals.size
        num_rejects, p_threshold = storey_bh(pvals, alpha_t, self.lmbda)

        self.R_sums[self.t] = self.R_total_sum
        self.R_sums[1:self.t] += num_rejects
        self.R_total_sum += num_rejects
        self.alphas[self.t] = alpha_t

        R_plus = 0
        for i, p in enumerate(pvals):
            pvals[i] = 0
            R_plus = max(R_plus, storey_bh(pvals, alpha_t, self.lmbda)[0])
            pvals[i] = p
        self.R_pluses[self.t] = R_plus

        self.t += 1
        #fdh = (self.alphas[1:self.t] * self.ks[1:self.t] * self.R_pluses[1:self.t] / (self.R_pluses[1:self.t] + self.R_sums[1:self.t])).sum()  # fdp hat
        return (pvals <= p_threshold).astype(float), 0, alpha_t, R_plus - num_rejects

    @staticmethod
    def _gamma(x, size):
        if size < 100:
            return gamma('poly', x)
        else:
            return gamma('half', x)


class Saffron(SingleTester):
    def __init__(self, alpha):
        self.alpha, self.w0 = alpha, alpha / 2

        # parameters for test level calculation
        self.lmbda = 0.5
        self.reject_times = np.zeros(128)
        self.candidates_since_rejects = np.zeros(128)
        self.rejects = 0
        self.t = 1

        # parameters for FDP hat calculation
        self.fdp_numerator = 0

    def test(self, pval):
        alpha_t = self.w0 * self._gamma(self.t - self.candidates_since_rejects[0])
        if self.rejects >= 1:
            alpha_t += (self.alpha - self.w0) * self._gamma(self.t - self.reject_times[1] - self.candidates_since_rejects[1])
            if self.rejects >= 2:
                alpha_t += self.alpha * self._gamma(self.t - self.reject_times[2:self.rejects+1] - self.candidates_since_rejects[2:self.rejects+1]).sum()
        alpha_t = min(self.lmbda, (1-self.lmbda) * alpha_t)

        if pval[0] <= self.lmbda:
            self.candidates_since_rejects[:self.rejects+1] += 1
        else:
            self.fdp_numerator += alpha_t / (1 - self.lmbda)

        if pval[0] <= alpha_t:  # rejection
            prediction = np.ones(1)
            self.rejects += 1
            if self.reject_times.size == self.rejects:
                self.reject_times = np.append(self.reject_times, np.zeros(self.rejects))
                self.candidates_since_rejects = np.append(self.candidates_since_rejects, np.zeros(self.rejects))
            self.reject_times[self.rejects] = self.t
        else:  # non-rejection
            prediction = np.zeros(1)

        self.t += 1
        fdh = self.fdp_numerator / max(self.rejects, 1)  # fdp hat
        return prediction, fdh, alpha_t, 0

    @staticmethod
    def _gamma(x):
        return gamma('poly', x)


#### PRDS ####


class BatchPrds(BatchTester):
    def __init__(self, alpha):
        self.alpha = alpha

        self.t = 1
        self.R_total_sum = 0

    def test(self, pvals):
        alpha_t = self.alpha * self._gamma(self.t) / pvals.size * (pvals.size + self.R_total_sum)
        num_rejects, p_threshold = bh(pvals, alpha_t)

        self.R_total_sum += num_rejects

        self.t += 1
        return (pvals <= p_threshold).astype(float), 0, alpha_t, 0

    @staticmethod
    def _gamma(x):
        return gamma('poly', x)


class Lond(SingleTester):
    def __init__(self, alpha):
        self.alpha = alpha

        self.t = 1
        self.R_total_sum = 0

    def test(self, pval):
        alpha_t = self.alpha * self._gamma(self.t) * (1 + self.R_total_sum)
        pred = np.ones(1) if pval <= alpha_t else np.zeros(1)

        self.R_total_sum += pred

        self.t += 1
        return pred, 0, alpha_t, 0

    @staticmethod
    def _gamma(x):
        return gamma('log', x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.05,
        help='Target FDR.')
    parser.add_argument('--n-trials', type=int, default=500,
        help='Number of trials to average over.')
    parser.add_argument('--n-hypos', type=int, default=3000,
        help='Length of hypotheses sequence.')
    parser.add_argument('--pi1s', type=float, nargs='+', default=np.append(np.arange(1, 10) / 100, np.arange(1, 6) / 10),
        help='Probabilities of non-nulls. Separate each pi1 value by a space.')
    parser.add_argument('--task', choices=('1S-Mean3', '2S-GaussianMean', '1S-PRDS'), default='1S-Mean3',
        help='Experimental setting.')
    parser.add_argument('--prds-size', type=int,
        help='Size of the each positively dependent set. Must be specified if 1S-PRDS is used.')
    parser.add_argument('--dir', type=str, default=timestamp(),
        help='Name of directory to save output in.')
    parser.add_argument('--real', type=str, default=None,
        help='Path to a pickled pandas dataframe of the p-vales for a real dataset. If None, uses the synthetic task.')
    parser.add_argument('--real-splits', type=int, default=100,
        help='Number of random splits of the real dataset.')
    parser.add_argument('--testers', choices=('online-nonadaptive', 'online-adaptive', 'nonadaptive', 'adaptive', 'batchbh', 'batchsbh', 'bh', 'sbh', 'prds-100', 'prds-1000'), required=True)
    parser.add_argument('--monotone', action='store_true', default=False,
        help='Calculate the percentage of trials that are monotone after setting a random null p-value to 0.')
    parser.add_argument('--n-cpus', type=int, default=-1,
        help='Number of cpus to use. Default is to use all cpus.')
    parser.add_argument('--rdiff', action='store_true', default=False,
        help='Plot histogram of R_t^+ - R_t')
    args = parser.parse_args()
    args.n_cpus = mp.cpu_count() if args.n_cpus == -1 else args.n_cpus
    OUT_DIR = Path('out', args.dir)
    if not OUT_DIR.is_dir():
        os.makedirs(OUT_DIR)
    with Path(OUT_DIR, 'args.txt').open('w') as f:
        f.write(str(args))
    print(args)
    if args.real:
        args.n_trials = args.real_splits
        args.pi1s = [0.]
        args.n_hypos = pd.read_pickle(args.real).xs('df0').shape[0]
    main(args)
