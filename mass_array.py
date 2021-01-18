import numpy
from publication_results import get_statistical_errors_central_fit
from model_definitions import K1, mPT_1loop
import sys


def get_masses(N, g, L, num_m):
    # Use the previous work to predict m_crit
    params = get_statistical_errors_central_fit(2)['params_central']
    alpha = params[0]
    beta = params[-2]
    nu = params[-1]

    m_crit = mPT_1loop(g, N) + g ** 2 * (alpha - beta * K1(g, N))

    # The number of multiples of m_c^2 * (gL)^1/nu
    span = 10

    x = g * L

    m_max = m_crit - span * x ** (-1 / nu) * m_crit
    m_min = m_crit + span * x ** (-1 / nu) * m_crit


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

    num_m = 20

    masses = get_masses(N, g, L, num_m)

    with open('masses_to_run_temp.txt', 'w') as f:
        s = ' '
        f.write(s.join(masses))
