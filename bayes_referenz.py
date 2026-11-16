"""Analytische Bayes-Referenz für das Generatormodell mit AWGN-Kanal.

Beide Klassen haben die Kovarianz diag(sigma^2 + sigma_n,j^2). Das Log-Likelihood-Verhältnis (LLR) ist daher linear in y,
    LLR(y) = w^T y + b,   w_j = dmu_j / (sigma^2 + sigma_n,j^2),
und klassenbedingt normalverteilt: LLR | DDoS ~ N(+D^2/2, D^2), LLR | Normal ~ N(-D^2/2, D^2)
mit D^2 = sum_j dmu_j^2 / (sigma^2 + sigma_n,j^2). Die Posterior-Schwelle p > t entspricht
    LLR > tau(t) = ln(t/(1-t)) - ln(pi/(1-pi)).

Genutzt von pr_ebene_plot.py, llr_signalraum_plot.py und entscheidungsfunktion_plot.py.
"""

import numpy as np
from scipy.stats import norm

from train_eval import ATTACK_PROB, MU_ATTACK, MU_NORMAL, SIGMA, SIGNAL_POWER

PI = float(ATTACK_PROB)


def _var(snr_db: float) -> np.ndarray:
    """Varianz je Merkmal der Beobachtung y: sigma^2 + sigma_n,j^2."""
    return float(SIGMA) ** 2 + SIGNAL_POWER.astype(np.float64) / 10.0 ** (snr_db / 10.0)


def mahalanobis_sq(snr_db: float) -> float:
    dmu = (MU_ATTACK - MU_NORMAL).astype(np.float64)
    return float(np.sum(dmu**2 / _var(snr_db)))


def tau(t):
    """LLR-Schwelle, die der Posterior-Schwelle p > t entspricht."""
    t = np.asarray(t, dtype=np.float64)
    return np.log(t / (1 - t)) - np.log(PI / (1 - PI))


def bayes_point(snr_db: float, t):
    """Precision, Recall, F1 des Bayes-Klassifikators mit Posterior-Schwelle t (Skalar oder Array)."""
    d2 = mahalanobis_sq(snr_db)
    d = np.sqrt(d2)
    th = tau(t)
    rec = norm.sf((th - d2 / 2) / d)
    fpr = norm.sf((th + d2 / 2) / d)
    prec = PI * rec / (PI * rec + (1 - PI) * fpr)
    f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def llr(y: np.ndarray, snr_db: float) -> np.ndarray:
    """Log-Likelihood-Verhältnis ln p(y|DDoS)/p(y|Normal) für Beobachtungen y (n x 8) bei bekanntem SNR."""
    var = _var(snr_db)
    mu_a, mu_n = MU_ATTACK.astype(np.float64), MU_NORMAL.astype(np.float64)
    w = (mu_a - mu_n) / var
    b = -np.sum((mu_a**2 - mu_n**2) / (2 * var))
    return np.asarray(y, dtype=np.float64) @ w + b


def posterior(y: np.ndarray, snr_db: float) -> np.ndarray:
    """Bayes-Posterior P(DDoS | y) bei bekanntem SNR und Angriffsanteil pi."""
    z = llr(y, snr_db) + np.log(PI / (1 - PI))
    return 1.0 / (1.0 + np.exp(-z))


def posterior_mischung(y: np.ndarray, snr_levels=(30, 20, 10, 5)) -> np.ndarray:
    """Bayes-Posterior P(DDoS | y) ohne Kenntnis des SNR, bei gleichverteilter Mischung der SNR-Stufen
    (so wie im Training gezogen). Klassendichte p(y | c) = Mittel der Gaußdichten über die Stufen."""
    from scipy.special import logsumexp

    y = np.asarray(y, dtype=np.float64)
    mu = {0: MU_NORMAL.astype(np.float64), 1: MU_ATTACK.astype(np.float64)}
    logp = {}
    for c in (0, 1):
        terms = []
        for s in snr_levels:
            var = _var(s)
            terms.append(-0.5 * np.sum((y - mu[c]) ** 2 / var + np.log(2 * np.pi * var), axis=1))
        logp[c] = logsumexp(terms, axis=0) - np.log(len(snr_levels))
    z = logp[1] - logp[0] + np.log(PI / (1 - PI))
    return 1.0 / (1.0 + np.exp(-z))


def huber_differenz(p):
    """Q_H(y,1) - Q_H(y,0) bei Regression mit Huber-Verlust (beta = 1) auf die Zweipunkt-Rewardverteilungen
    {+1, -1} (Aktion DDoS) und {-2, +0,5} (Aktion Normal) mit Angriffswahrscheinlichkeit p.
    Geschlossene Form: (3,5p - 1,5)/(1-p) für p < 1/2 und (5p - 2)/p für p > 1/2; Nullstelle 3/7."""
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p < 0.5, (3.5 * p - 1.5) / (1 - p), (5 * p - 2) / p)
