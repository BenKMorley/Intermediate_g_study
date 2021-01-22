import numpy
from publication_results import get_statistical_errors_central_fit
from model_definitions import K1, mPT_1loop
import sys


def get_masses(N, g, L, num_m, span=10):
    """
        span is the number of multiples of g^2 * (gL)^1/nu
    """
    # Use the previous work to predict m_crit
    params = get_statistical_errors_central_fit(N)['params_central']
    alpha = params[0]
    beta = params[-2]
    nu = params[-1]

    m_crit = mPT_1loop(g, N) + g ** 2 * (alpha - beta * K1(g, N))

    x = g * L

    m_max = m_crit - span * x ** (-1 / nu) * g ** 2
    m_min = m_crit + span * x ** (-1 / nu) * g ** 2


    # Save the masses at the 5 dp. level for use in a bash script
    masses = numpy.linspace(m_min, m_max, num_m)

    masses = list(masses)
    masses = [f"{mass:.5f}" for mass in masses]

    return masses


if __name__ == "__main__":
    N, g, L = sys.argv[1:]
    N = int(N)
    g = float(g)
    L = int(L)
    span = 10

    num_m = 20

    masses = get_masses(N, g, L, num_m, span=span)

    with open('masses_to_run_temp.txt', 'w') as f:
        s = ' '
        f.write(s.join(masses))
