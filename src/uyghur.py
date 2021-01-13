from numpy.random import rand
from math import log, inf
from pTSL import pTSL
from scipy.optimize import minimize

import csv

# Starting probs

OUTPUT_FILE = 'predicted_probabilities_uyghur.csv'
SAMPLING_RESULTS = 'parameter_sampling.csv'

PROB_NAMES = [
    'F',
    'B',
    'N',
    'K',
    'Q',
    'C',
    'Sf',
    'Sb'
]

K_FACTORS = [
    ('F', 'Sb'),
    ('B', 'Sf'),

    ('N', 'Sf'),
    ('#', 'Sf'),

    ('K', 'Sb'),
    ('Q', 'Sf')
]

BOUNDS = [
    (1, 1), # F
    (1, 1), # B
    (0, 1), # N
    (0, 1), # K
    (0, 1), # Q
    (0, 1), # C
    (1, 1), # Sf
    (1, 1), # Sb
]

starting_probs = [
    1, # F
    1, # B
    0.5, # N
    0.5, # K
    0.5, # Q
    0.5, # C
    1, # Sf
    1, # Sb
]

def evaluate(probs, verbose=False, output_file=None):
    prob_dict = dict(zip(PROB_NAMES, probs))
    grammar = pTSL(prob_dict, K_FACTORS)

    if output_file:
        open(output_file, 'w').close()

    loglik = 0

    for i, (form1, form2) in enumerate(TEST_PAIRS):
        front_form_p = grammar.val(form1)
        back_form_p = grammar.val(form2)
        back_prob = back_form_p / (back_form_p + front_form_p)
        front_prob = 1 - back_prob

        if output_file:
            with open(output_file, 'a') as f:
                f.write(','.join([''.join(form1[:-1]), str(back_prob)]) + '\n')

        if back_prob > 0:
            back_log_prob = log(back_prob, 10)
        else:
            back_log_prob = -10000000

        if front_prob > 0:
            front_log_prob = log(front_prob, 10)
        else:
            front_log_prob = -10000000

        front_count = EXPECTED_FREQUENCIES[2*i]
        back_count = EXPECTED_FREQUENCIES[2*i + 1]

        loglik += front_count * front_log_prob
        loglik += back_count * back_log_prob

        if verbose:
            print('{}:{}'.format(form1, front_form_p))
            print('{}:{}'.format(form2, back_form_p))
            print('Predicted back: {}'.format(back_prob))
            print('Predicted front: {}'.format(front_prob))
            print('Predicted back log: {}'.format(back_log_prob))
            print('Predicted front log: {}'.format(front_log_prob))
            print('Observed back: {}'.format(EXPECTED_FREQUENCIES[i+1]))
            print('Observed front: {}'.format(EXPECTED_FREQUENCIES[i]))
            print('Back lik: {}'.format(back_count * back_log_prob))
            print('Front lik: {}'.format(front_count * front_log_prob))

    return -loglik

TEST_PAIRS = (
    (
        ('C', 'F', 'C', 'Sf'),
        ('C', 'F', 'C', 'Sb')
    ),
    (
        ('C', 'F', 'C', 'N', 'C', 'Sf'),
        ('C', 'F', 'C', 'N', 'C', 'Sb'),
    ),
    (
        ('C', 'F', 'C', 'N', 'C', 'N', 'C', 'Sf'),
        ('C', 'F', 'C', 'N', 'C', 'N', 'C', 'Sb'),
    ),
    (
        ('C', 'F', 'Q', 'Sf'),
        ('C', 'F', 'Q', 'Sb'),
    ),
    (
        ('C', 'F', 'C', 'N', 'Q', 'Sf'),
        ('C', 'F', 'C', 'N', 'Q', 'Sb'),
    ),
    (
        ('C', 'F', 'C', 'N', 'C', 'N', 'Q', 'Sf'),
        ('C', 'F', 'C', 'N', 'C', 'N', 'Q', 'Sb'),
    ),
    (
        ('C', 'B', 'C', 'Sf'),
        ('C', 'B', 'C', 'Sb'),
    ),
    (
        ('C', 'B', 'C', 'N', 'C', 'Sf'),
        ('C', 'B', 'C', 'N', 'C', 'Sb'),
    ),
    (
        ('C', 'B', 'C', 'N', 'C', 'N', 'C', 'Sf'),
        ('C', 'B', 'C', 'N', 'C', 'N', 'C', 'Sb'),
    ),
    (
        ('C', 'B', 'K', 'Sf'),
        ('C', 'B', 'K', 'Sb'),
    ),
    (
        ('C', 'B', 'C', 'N', 'K', 'Sf'),
        ('C', 'B', 'C', 'N', 'K', 'Sb'),
    ),
    (
        ('C', 'B', 'C', 'N', 'C', 'N', 'K', 'Sf'),
        ('C', 'B', 'C', 'N', 'C', 'N', 'K', 'Sb'),
    ),
    # (
    #     ('C', 'N', 'C', 'Sf'),
    #     ('C', 'N', 'C', 'Sb'),
    # ),
    # (
    #     ('C', 'N', 'C', 'N', 'C', 'Sf'),
    #     ('C', 'N', 'C', 'N', 'C', 'Sb'),
    # ),
    # (
    #     ('C', 'N', 'C', 'N', 'C', 'N', 'C', 'Sf'),
    #     ('C', 'N', 'C', 'N', 'C', 'N', 'C', 'Sb'),
    # )
)

EXPECTED_FREQUENCIES = [
    81, # F Sf
    3, # F Sb
    68, # F N Sf
    15, # F N Sb
    32, # F N N Sf
    52, # F N N Sb

    64, # F Q Sf
    20, # F Q Sb
    52, # F N Q Sf
    32, # F N Q Sb
    20, # F N N Q Sf
    64, # F N N Q Sb

    7, # B Sf
    77, # B Sb
    16, # B N Sf
    68, # B N Sb
    11, # B N N Sf
    73, # B N N Sb

    7, # B K Sf
    77, # B K Sb
    6, # B N K Sf
    78, # B N K Sb
    14, # B N N K Sf
    69, # B N N K Sb

    # 14, # C N C Sf
    # 69, # C N C Sb

    # 15, # C N C N C Sf
    # 69, # C N C N C Sb

    # 13, # C N C N C N C Sf
    # 71, # C N C N C N C Sb
]

results = dict()
best_lik = inf
best_params = None
best_results = None
worst_lik = -inf
worst_params = None
worst_results = None

with open(SAMPLING_RESULTS, 'w') as f:
    writer = csv.writer(f)
    for i in range(0, 1):
        print("Random start {}".format(i))
        random_start = rand(len(PROB_NAMES))
        res = minimize(evaluate, random_start, bounds=BOUNDS, method='L-BFGS-B')
        writer.writerow([
            ' '.join(["{}:{}".format(x, y) for (x, y) in zip(PROB_NAMES, starting_probs)]),
            ' '.join(["{}:{}".format(x, y) for (x, y) in zip(PROB_NAMES, res.x)]),
            res.fun
        ])

        if res.fun < best_lik:
            best_lik = res.fun
            best_params = random_start
            best_results = res

        if res.fun > worst_lik:
            worst_lik = res.fun
            worst_params = random_start
            worst_results = res

print("BEST")
print(best_lik)
print(best_params)
print(best_results)

print("WORST")
print(worst_lik)
print(worst_params)
print(worst_results)

evaluate(best_results.x, output_file = OUTPUT_FILE)

p1 = [
    1, # F
    1, # B
    0.27, # N
    0.07, # K
    0.27, # Q
    0.07, # C
    1, # Sf
    1, # Sb
]
evaluate(p1, output_file = 'uyghur_CK007_QN027.csv')

p2 = [
    1, # F
    1, # B
    0.27, # N
    0.27, # K
    0.27, # Q
    0.27, # C
    1, # Sf
    1, # Sb
]

evaluate(p2, output_file = 'uyghur_CK027_QN027.csv')

p3 = [
    1, # F
    1, # B
    0.07, # N
    0.07, # K
    0.07, # Q
    0.07, # C
    1, # Sf
    1, # Sb
]

evaluate(p3, output_file = 'uyghur_CK007_QN007.csv')

p4 = [
    1, # F
    1, # B
    0.07, # N
    0.27, # K
    0.07, # Q
    0.27, # C
    1, # Sf
    1, # Sb
]

evaluate(p4, output_file = 'uyghur_CK027_QN007.csv')