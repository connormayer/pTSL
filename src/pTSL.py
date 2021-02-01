import argparse
import csv
from scipy.optimize import minimize
from numpy.random import rand


class pTSL():
    '''
        probs is the set of prob projection, factors is the strictly k-local grammar
        probs is a dictionary e.g. (a : 0.55)
        factors is a grammar list of tuple containing each char as element e.g. [(ph,a),(sh,b),(a,c)]
    '''
    def __init__(self, probs, factors):
        self.probs = probs
        self.factors = factors

    def get_kfactors(self, string):
        # get k based on G. The length of the longest element is k (e.g. 2 for [(a,a),(a,b),(a,c)])
        k = max([len(x) for x in self.factors])
        # pad the string as ###string####
        padded_string = tuple((k-1) * ['#'] + string + (k-1) * ['#'])
        kfactors = []
        for i in range(0, len(padded_string) - k + 1):
            kfactors.append(padded_string[i:i+k])
        # return a list of k-factors (list of each 'window' to look at) based on G
        return kfactors

    def val(self, string):
        result = self.get_projections(string)
        return self.val_helper(result)

    def val_helper(self, projections):
        # all substring projections
        # if verbose, show results, too
        total_prob = 0
        for proj, val in projections:
            k_factors = self.get_kfactors(proj)
            # if any substring contains the forbidden grammar, do not count toward the total probability
            if not any([x in k_factors for x in self.factors]):
                total_prob += val
        return total_prob

    def get_projections(self, string):
        if not string:
            return [([], 1)]

        first, *rest = string
        # if a symbol is in the probs
        if first in self.probs.keys():
            prob = self.probs[first]
        # if a symbol is not in the probs, it is always 0 probability
        else:
            prob = 0
        projections = self.get_projections(rest)

        new_projections = []
        for proj, val in projections:
            new_projections.append(([first] + proj, prob * val))
            new_projections.append((proj, (1 - prob) * val))
        return new_projections

    '''
        read the corpus file into data and output the conditional probabilities fore each input
        NOTE: unused for now. Maybe useful when implementing GUI?
    '''
    def val_corpus(self, corpus_file):
        return self.val_corpus_helper(corpus_file)

    def val_corpus_helper(self, corpus_file):
        # read file into data
        corpus_data = self.read_corpus_file(corpus_file)
        cond_probs = []
        input_strings = []
        for string in corpus_data:
            cond_probs.append(self.val(string))
            input_strings.append(' '.join(map(str, string)))
        return dict(zip(input_strings, cond_probs))

    def print_conditional_probabilities(self, corpus_file, verbose=False):
        # read file into data
        corpus_data = self.read_corpus_file(corpus_file)
        for index, string in enumerate(corpus_data):
            projections = self.get_projections(string)
            result = self.val_helper(projections)
            print("Input {}: {}".format(index+1, ' '.join(map(str, string))))
            print("Input {} Probability: {}".format(index+1, round(result, 4)))
            if verbose:
                print("\nProjection Probabilities for input {}".format(index+1))
                for sub, value in projections:
                    if not value:
                        continue
                    print([' '.join(map(str, sub))], ': ', round(value, 3))
                print()

    def evaluate_proj(self, random_proj, param, corpus_probs):
        # return sum of sqaured errors

        for i, t in enumerate(param):
            self.probs[t] = random_proj[i]

        sse = 0
        for string, y in corpus_probs:
            sse += (self.val(string) - y)**2

        return sse

    def train(self, corpus_file):
        # [[['a','b'],1],[]]
        corpus_probs = self.read_corpus_file(corpus_file, True)
        # get list of tiers
        # from initialized ptsl (self.probs and self.factor) and corpus data, create complete list of alphabet (no duplicates)
        param = {x for sublist in corpus_probs for x in sublist[0]}
        # remove fixed alphabets
        param = list(param - {x for x in self.probs})

        # create bounds
        # instead of limiting bound for fixed value, I removed it completely from the parameter
        bounds = [(0, 1) for i in range(len(param))]

        # bounds = []
        # for t in param:
        #     if t in self.probs:
        #         bounds.append((self.probs[t], self.probs[t]))
        #     else:
        #         bounds.append((0, 1))

        # randomly initialize parameter - this will be the input
        random_proj = rand(len(param))
        # run the minimize function
        proj_res = minimize(self.evaluate_proj,
                            random_proj,
                            bounds=bounds,
                            method='L-BFGS-B',
                            args=(param, corpus_probs))

        optimal_projection = dict(zip(param, proj_res.x))
        self.probs.update(optimal_projection)

    def print_projections(self):
        for key in sorted(self.probs.keys()):
            print("{}: {}".format(key, round(self.probs[key], 4)))

    ''' 
    initialize pTSL based on the given path to files (tier, prob, grammar)
    '''
    @classmethod
    def from_file(cls, tier_file, factors_file, verbose=False):
        '''
            An alternative constructor that creates a pTSL object based on
            the contents of input files. The files should have the following formats:

            TIER_FILE
            <symbol 1> <projection 1>
            <symbol 2> <projection 2>
            ...
            <symbol n> <projection n>

            FACTORS_FILE
            <factor 1>
            <factor 2>
            ...
            <factor n>


            <factor i> is a segment separated by spaces.
            Segments do not need to be single characters.

            Input:
                tier_file: A string specifying where to find the tier file.
                factors_file: a string specifying where to find the factor file
        '''

        probs = {}
        with open(tier_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) == 1:
                    probs[row[0]] = 0
                else:
                    probs[row[0]] = float(row[1])

        factors = []
        with open(factors_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                factors.append(tuple(row[0].split()))

        return pTSL(probs, factors)

    @staticmethod
    def read_corpus_file(corpus_file, train=False):
        '''
            Read corpus data from a file
            The file should have the following formats:

            <string 1> <probability 1>
            <string 2> <probability 1>
            ...
            <string n> <probability 1>


            <string i> is a segment separated by spaces.
            Segments do not need to be single characters.

            probabilities are optional. They are used for the purpose of learning projection probabilities.

            Input:
                corpus_file: A string specifying where to find the corpus file
                train:
        '''
        corpus_data = []
        with open(corpus_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            if train:
                for row in csv_reader:
                    corpus_data.append([row[0].split(), float(row[1])])
            else:
                for row in csv_reader:
                    corpus_data.append(row[0].split())

        return corpus_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the probability assigned to input corpus."
    )

    parser.add_argument(
        'input_file', type=str, action='store', nargs=3,
        help='Paths to the tier-projection file, factors file, and input corpus file.'
    )

    parser.add_argument(
        '--train', action='store_true',
        help='The path to the file corpus with probabilities.'
    )

    parser.add_argument(
        '--verbose', action='store_true',
        help='Prints probabilities for each possible projection.'
    )

    # parser.add_argument(
    #     '--output_file', type=str, default=None,
    #     help='Path to csv file to save probabilities.',
    # )

    args = parser.parse_args()
    input_tier = args.input_file[0]
    input_factors = args.input_file[1]
    input_corpus = args.input_file[2]

    ptsl = pTSL.from_file(input_tier, input_factors)
    if args.train:
        ptsl.train(input_corpus)
        ptsl.print_projections()

        ptsl.print_conditional_probabilities(input_corpus, args.verbose)

    else:
        ptsl.print_conditional_probabilities(input_corpus, args.verbose)

