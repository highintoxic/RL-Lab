"""
FAST TEST VERSION
Naive vs MEA (PAC Best Arm Identification)
Small values for quick execution
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time


# ============================================================
# Generate Bandit
# ============================================================

def generate_bandit(K):
    return np.random.uniform(0, 1, K)


# ============================================================
# Naive Algorithm
# ============================================================

def naive_algorithm(mu, epsilon, delta):
    K = len(mu)

    m = math.ceil((2 / (epsilon ** 2)) * math.log((2 * K) / delta))

    samples = np.random.binomial(1, mu, (m, K))
    empirical_means = np.mean(samples, axis=0)

    chosen_arm = np.argmax(empirical_means)
    total_samples = m * K

    return chosen_arm, empirical_means, total_samples


# ============================================================
# Median Elimination
# ============================================================

def median_elimination(mu, epsilon, delta):
    S = np.arange(len(mu))
    eps_l = epsilon / 4
    delta_l = delta / 2
    total_samples = 0
    phase_history = []

    while len(S) > 1:

        phase_history.append(len(S))

        m = math.ceil((4 / (eps_l ** 2)) * math.log(3 / delta_l))

        samples = np.random.binomial(1, mu[S], (m, len(S)))
        empirical_means = np.mean(samples, axis=0)

        total_samples += m * len(S)

        median_value = np.median(empirical_means)
        S = S[empirical_means >= median_value]

        eps_l *= 0.75
        delta_l *= 0.5

    return S[0], total_samples, phase_history


# ============================================================
# 1️⃣ Success Rate Test
# ============================================================

def experiment_success_rate():
    print("Running Success Rate Test...")

    K = 200
    epsilon = 0.1
    delta = 0.1

    naive_success = 0
    mea_success = 0

    for _ in range(20):   # small runs for testing
        mu = generate_bandit(K)
        optimal_value = np.max(mu)

        arm_n, _, _ = naive_algorithm(mu, epsilon, delta)
        if mu[arm_n] >= optimal_value - epsilon:
            naive_success += 1

        arm_m, _, _ = median_elimination(mu, epsilon, delta)
        if mu[arm_m] >= optimal_value - epsilon:
            mea_success += 1

    print("Naive ε-optimal count:", naive_success)
    print("MEA ε-optimal count:", mea_success)
    print()


# ============================================================
# 2️⃣ Sample Complexity vs K
# ============================================================

def experiment_sample_complexity():
    print("Running Sample Complexity Test...")

    epsilon = 0.1
    delta = 0.1

    Ks = [100, 300, 500, 700, 1000]

    naive_samples = []
    mea_samples = []

    for K in Ks:
        mu = generate_bandit(K)

        _, _, n_samples = naive_algorithm(mu, epsilon, delta)
        _, m_samples, _ = median_elimination(mu, epsilon, delta)

        naive_samples.append(n_samples)
        mea_samples.append(m_samples)

        print("K =", K, "done")

    plt.figure()
    plt.plot(Ks, naive_samples, label="Naive")
    plt.plot(Ks, mea_samples, label="MEA")
    plt.xlabel("Number of Arms (K)")
    plt.ylabel("Sample Complexity")
    plt.title("Sample Complexity (Test Version)")
    plt.legend()
    plt.show()


# ============================================================
# 3️⃣ Empirical Error Test
# ============================================================

def experiment_empirical_error():
    print("Running Empirical Error Test...")

    K = 2000
    mu = generate_bandit(K)
    optimal_value = np.max(mu)

    epsilons = [0.2, 0.1, 0.05]
    delta_fixed = 0.1

    naive_errors = []
    mea_errors = []

    for epsilon in epsilons:
        arm_n, _, _ = naive_algorithm(mu, epsilon, delta_fixed)
        arm_m, _, _ = median_elimination(mu, epsilon, delta_fixed)

        naive_errors.append(abs(mu[arm_n] - optimal_value))
        mea_errors.append(abs(mu[arm_m] - optimal_value))

    plt.figure()
    plt.plot(epsilons, naive_errors, label="Naive")
    plt.plot(epsilons, mea_errors, label="MEA")
    plt.xlabel("Epsilon")
    plt.ylabel("|μ̂ - μ*|")
    plt.title("Error vs Epsilon (Test Version)")
    plt.legend()
    plt.show()


# ============================================================
# 4️⃣ Phase History
# ============================================================

def experiment_phase_history():
    print("Running MEA Phase History Test...")

    K = 200
    epsilon = 0.1
    delta = 0.1

    mu = generate_bandit(K)
    _, _, phase_history = median_elimination(mu, epsilon, delta)

    plt.figure()
    plt.plot(range(len(phase_history)), phase_history)
    plt.xlabel("Phase")
    plt.ylabel("Remaining Arms")
    plt.title("MEA Phase History (Test Version)")
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    start = time.perf_counter()

    experiment_success_rate()
    experiment_sample_complexity()
    experiment_empirical_error()
    experiment_phase_history()

    print("Total Execution Time:",
          time.perf_counter() - start)