import random
import logging
import math

def generate_data(pA: float, pB: float, pC: float, N: int):
    """
    Generate N trials of the three-coin experiment.
    Each trial: toss coin A (latent), then toss coin B or C based on A's outcome.

    Args:
        pA: probability of heads for coin A
        pB: probability of heads for coin B
        pC: probability of heads for coin C
        N: number of trials

    Returns:
        List of tuples (a, o) where:
          a (int): 1 if coin A was heads, else 0
          o (int): 1 if chosen coin (B or C) was heads, else 0
    """
    data = []
    for _ in range(N):
        a = 1 if random.random() < pA else 0
        if a:
            o = 1 if random.random() < pB else 0
        else:
            o = 1 if random.random() < pC else 0
        data.append((a, o))
    return data


def em_solver(observations: list, init_pA: float, init_pB: float, init_pC: float,
              max_iter: int = 100, tol: float = 1e-6, log_file: str = 'em_log.txt'):
    """
    EM algorithm for the three-coin problem.

    Args:
        observations: list of tuples (_, o), only o used in EM
        init_pA, init_pB, init_pC: initial parameter guesses
        max_iter: maximum number of EM iterations
        tol: convergence tolerance for parameter changes
        log_file: path to write iteration logs

    Returns:
        Tuple (pA, pB, pC): estimated parameters
    """
    # Set up logging
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s')

    pA, pB, pC = init_pA, init_pB, init_pC
    N = len(observations)

    for t in range(1, max_iter + 1):
        # E-step: compute responsibilities q_t^i(1)
        q1 = []
        for _, o in observations:
            like1 = pA * (pB**o) * ((1 - pB)**(1 - o))
            like0 = (1 - pA) * (pC**o) * ((1 - pC)**(1 - o))
            denom = like1 + like0
            q1.append(like1 / denom)

        # M-step: update parameters
        pA_new = sum(q1) / N
        sum_q1 = sum(q1)
        sum_q0 = N - sum_q1
        pB_new = sum(q1[i] * observations[i][1] for i in range(N)) / sum_q1
        pC_new = sum((1 - q1[i]) * observations[i][1] for i in range(N)) / sum_q0

        # Compute log-likelihood under new parameters
        loglik = 0.0
        for _, o in observations:
            like = pA_new * (pB_new**o) * ((1 - pB_new)**(1 - o))
            like += (1 - pA_new) * (pC_new**o) * ((1 - pC_new)**(1 - o))
            loglik += math.log(like)
        loglik /= N
        # Log iteration details
        logging.info(f"Iter {t}: loglik={loglik:.6f}, pA={pA_new:.6f}, pB={pB_new:.6f}, pC={pC_new:.6f}")

        # Check convergence
        if max(abs(pA_new - pA), abs(pB_new - pB), abs(pC_new - pC)) < tol:
            pA, pB, pC = pA_new, pB_new, pC_new
            break

        pA, pB, pC = pA_new, pB_new, pC_new

    return pA, pB, pC


if __name__ == "__main__":
    # Example usage:
    true_pA, true_pB, true_pC = 0.6, 0.5, 0.4
    N = 1000
    data = generate_data(true_pA, true_pB, true_pC, N)

    # Discard latent 'a' when running EM
    observations = [(None, o) for _, o in data]

    init = (0.9, 0.3, 0.8)
    est_pA, est_pB, est_pC = em_solver(observations, *init)
    print(f"Estimated pA={est_pA:.4f}, pB={est_pB:.4f}, pC={est_pC:.4f}")
