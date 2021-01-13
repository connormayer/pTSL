from math import log, inf
from numpy.random import rand
from pTSL import pTSL
from scipy.optimize import minimize

import csv

RATING_OUTPUT_FILE = 'predicted_ratings_hungarian.csv'
FREQ_OUTPUT_FILE = 'predicted_freqs_hungarian.csv'

DATA_FILE = '../data/hungarian_data.csv'

RATING_SAMPLING_RESULTS = 'hungarian_rating_parameter_sampling.csv'
FREQ_SAMPLING_RESULTS = 'hungarian_freq_parameter_sampling.csv'

PROB_NAMES = [
    'B',
    'I',
    'e:',
    'e',
    'Sf',
    'Sb'
]

K_FACTORS = [
    ('B', 'Sf'),
    #('#', 'Sb'),
    ('I', 'Sb'),
    ('e:', 'Sb'),
    ('e', 'Sb')
]

BOUNDS = [
    (1, 1), # B
    (0, 1), # I
    (0, 1), # é
    (0, 1), # e
    (1, 1), # Sf
    (1, 1), # Sb
]

def evaluate_ratings(probs, data, headers, verbose=False, output_file=None):
    prob_dict = dict(zip(PROB_NAMES, probs))
    grammar = pTSL(prob_dict, K_FACTORS)

    sse = 0

    if output_file:
        open(output_file, 'w').close()

    for i, row in enumerate(data):
        front_form = row[headers.index('front.form')].split(' ')
        back_form = row[headers.index('back.form')].split(' ')
        front_form_p = grammar.val(front_form)
        back_form_p = grammar.val(back_form)

        front_obs = float(row[headers.index('front.rating')])
        back_obs = float(row[headers.index('back.rating')])

        sse += (front_form_p - front_obs)**2
        sse += (back_form_p - back_obs)**2

        if output_file:
            with open(output_file, 'a') as f:
                f.write(','.join([''.join(front_form[:-1]), str(front_form_p), front_form[-1][-1]]) + '\n')
                f.write(','.join([''.join(back_form[:-1]), str(back_form_p), back_form[-1][-1]]) + '\n')

        if verbose:
            print('{}:{}'.format(form1, front_form_p))
            print('{}:{}'.format(form2, back_form_p))

    return sse

def evaluate_mle(probs, data, headers, verbose=False, output_file=None):
    prob_dict = dict(zip(PROB_NAMES, probs))
    grammar = pTSL(prob_dict, K_FACTORS)

    if output_file:
        open(output_file, 'w').close()

    loglik = 0

    for i, row in enumerate(data):
        front_form = row[headers.index('front.form')].split(' ')
        back_form = row[headers.index('back.form')].split(' ')
        front_form_p = grammar.val(front_form)
        back_form_p = grammar.val(back_form)

        back_prob = back_form_p / (back_form_p + front_form_p)
        front_prob = 1 - back_prob

        if output_file:
            with open(output_file, 'a') as f:
                f.write(','.join([''.join(front_form[:-1]), str(back_prob)]) + '\n')

        if back_prob > 0:
            back_log_prob = log(back_prob, 10)
        else:
            back_log_prob = -10000000

        if front_prob > 0:
            front_log_prob = log(front_prob, 10)
        else:
            front_log_prob = -10000000

        front_count = float(row[headers.index('front.total')])
        back_count = float(row[headers.index('back.total')])

        loglik += front_count * front_log_prob
        loglik += back_count * back_log_prob

        if verbose:
            print('{}:{}'.format(front_form, front_form_p))
            print('{}:{}'.format(back_form, back_form_p))
            print('Predicted back: {}'.format(back_prob))
            print('Predicted front: {}'.format(front_prob))
            print('Predicted back log: {}'.format(back_log_prob))
            print('Predicted front log: {}'.format(front_log_prob))
            print('Observed back: {}'.format(EXPECTED_FREQUENCIES[i+1]))
            print('Observed front: {}'.format(EXPECTED_FREQUENCIES[i]))
            print('Back lik: {}'.format(back_count * back_log_prob))
            print('Front lik: {}'.format(front_count * front_log_prob))

    return -loglik


# TEST_PAIRS = (
#     (
#         ('e:', 'Sf'),
#         ('e:', 'Sb')
#     ),
#     (
#         ('B', 'e:', 'Sf'),
#         ('B', 'e:', 'Sb')
#     ),
#     (
#         ('B', 'e:', 'e:', 'Sf'),
#         ('B', 'e:', 'e:', 'Sb')
#     ),
#     (
#         ('B', 'e:', 'ɛ', 'Sf'),
#         ('B', 'e:', 'ɛ', 'Sb')
#     ),
#     (
#         ('B', 'e:', 'I', 'Sf'),
#         ('B', 'e:', 'I', 'Sb')
#     ),
#     (
#         ('B', 'B', 'e:', 'Sf'),
#         ('B', 'B', 'e:', 'Sb')
#     ),
#     (
#         ('B', 'B', 'ɛ', 'Sf'),
#         ('B', 'B', 'ɛ', 'Sb')
#     ),
#     (
#         ('B', 'B', 'I', 'Sf'),
#         ('B', 'B', 'I', 'Sb')
#     ),
#     (
#         ('B', 'ɛ', 'Sf'),
#         ('B', 'ɛ', 'Sb')
#     ),
#     (
#         ('B', 'ɛ', 'e:', 'Sf'),
#         ('B', 'ɛ', 'e:', 'Sb')
#     ),
#     (
#         ('B', 'ɛ', 'ɛ', 'Sf'),
#         ('B', 'ɛ', 'ɛ', 'Sb')
#     ),
#     (
#         ('B', 'ɛ', 'I', 'Sf'),
#         ('B', 'ɛ', 'I', 'Sb')
#     ),
#     (
#         ('B', 'I', 'Sf'),
#         ('B', 'I', 'Sb')
#     ),
#     (
#         ('B', 'I', 'e:', 'Sf'),
#         ('B', 'I', 'e:', 'Sb')
#     ),
#     (
#         ('B', 'I', 'ɛ', 'Sf'),
#         ('B', 'I', 'ɛ', 'Sb')
#     ),
#     (
#         ('B', 'I', 'I', 'Sf'),
#         ('B', 'I', 'I', 'Sb')
#     ),
#     (
#         ('ɛ', 'Sf'),
#         ('ɛ', 'Sb')
#     ),
#     (
#         ('I', 'Sf'),
#         ('I', 'Sb')
#     ),

# )

# EXPECTED_RATINGS = [
#     0.93518519,  # E Sf
#     0.10300926,  # E Sb

#     0.57251082,  # B E Sf
#     0.53138528,  # B E Sb

#     0.74242424,  # B E E Sf
#     0.25757576,  # B E E Sb

#     0.91333333,  # B E e Sf
#     0.11333333,  # B E e Sb

#     0.85964912,  # B E I Sf
#     0.27192982,  # B E I Sb

#     0.65000000,  # B B E Sf
#     0.46250000,  # B B E Sb

#     0.78020833 , # B B e Sf
#     0.28541667,  # B B e Sb

#     0.47196262, # B B I Sf
#     0.62461059, # B B I Sb

#     0.76855124 , # B e Sf
#     0.31861013 , # B e Sb

#     0.81547619 , # B e E Sf
#     0.23214286 , # B e E Sb

#     0.87853107 , # B e e Sf
#     0.20338983, # B e e Sb

#     0.79279279 , # B e I Sf
#     0.26576577 , # B e I Sb

#     0.40842491, # B I Sf
#     0.69780220,  # B I Sb

#     0.84126984 , # B I E Sf
#     0.19841270, # B I E Sb

#     0.90740741 , # B I e Sf
#     0.18518519, # B I e Sb

#     0.64912281 , # B I I Sf
#     0.48245614 , # B I I Sb

#     0.93535826 , # e Sf
#     0.08722741,  # e Sb

#     0.90601504,  # I Sf
#     0.15664160,  # I Sb
# ]

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

headers = data[0]
data = data[1:]

best_rating_lik = inf
best_rating_params = None
best_rating_results = None
worst_rating_lik = -inf
worst_rating_params = None
worst_rating_results = None

with open(RATING_SAMPLING_RESULTS, 'w') as f:
    writer = csv.writer(f)
    for i in range(0, 1):
        print("Random start {}".format(i))
        random_start = rand(len(PROB_NAMES))
        rating_res = minimize(evaluate_ratings, random_start, bounds=BOUNDS, method='L-BFGS-B', args=(data, headers))

        writer.writerow([
            ' '.join(["{}:{}".format(x, y) for (x, y) in zip(PROB_NAMES, random_start)]),
            ' '.join(["{}:{}".format(x, y) for (x, y) in zip(PROB_NAMES, rating_res.x)]),
            rating_res.fun
        ])

        if rating_res.fun < best_rating_lik:
            best_rating_lik = rating_res.fun
            best_rating_params = random_start
            best_rating_results = rating_res

        if rating_res.fun > worst_rating_lik:
            worst_rating_lik = rating_res.fun
            worst_rating_params = random_start
            worst_rating_results = rating_res

print("BEST RATING")
print(best_rating_lik)
print(best_rating_params)
print(best_rating_results)

print("WORST RATING")
print(worst_rating_lik)
print(worst_rating_params)
print(worst_rating_results)

best_freq_lik = inf
best_freq_params = None
best_freq_results = None
worst_freq_lik = -inf
worst_freq_params = None
worst_freq_results = None

with open(FREQ_SAMPLING_RESULTS, 'w') as f:
    writer = csv.writer(f)

    for i in range(0, 1):
        print("Random start {}".format(i))
        random_start = rand(len(PROB_NAMES))
        mle_res = minimize(evaluate_mle, random_start, bounds=BOUNDS, method='L-BFGS-B', args=(data, headers))

        writer.writerow([
            ' '.join(["{}:{}".format(x, y) for (x, y) in zip(PROB_NAMES, random_start)]),
            ' '.join(["{}:{}".format(x, y) for (x, y) in zip(PROB_NAMES, mle_res.x)]),
            mle_res.fun
        ])
        if mle_res.fun < best_freq_lik:
            best_freq_lik = mle_res.fun
            best_freq_params = random_start
            best_freq_results = mle_res

        if mle_res.fun > worst_freq_lik:
            worst_freq_lik = mle_res.fun
            worst_freq_params = random_start
            worst_freq_results = mle_res

print("BEST FREQ")
print(best_freq_lik)
print(best_freq_params)
print(best_freq_results)

print("WORST FREQ")
print(worst_freq_lik)
print(worst_freq_params)
print(worst_freq_results)

print(evaluate_ratings(best_rating_results.x, data, headers, output_file = 'hungarian_predicted_ratings.csv'))
print(evaluate_mle(best_freq_results.x, data, headers, output_file = 'hungarian_predicted_freqs.csv'))

print(evaluate_ratings(best_freq_results.x, data, headers, output_file = 'hungarian_predicted_ratings_from_mle.csv'))
print(evaluate_mle(best_rating_results.x, data, headers, output_file = 'hungarian_predicted_freqs_from_ratings.csv'))

import pdb; pdb.set_trace()