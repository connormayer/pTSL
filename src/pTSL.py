import argparse
import csv
from scipy.optimize import minimize
from numpy.random import rand
import TSL
import warnings

DEFAULT_PROB_FILENAME = "../output/conditional_probabilities.csv"
DEFAULT_PROJECTION_FILENAME = "../output/projections.csv"
DEFAULT_TIER_PROB_FILENAME = "../output/predicted_tier_probs.csv"
DEFAULT_TSL_TIER_FILENAME = "../output/tsl_tier.csv"
DEFAULT_TSL_FACTOR_FILENAME = "../output/tsl_factor.csv"


class pTSL:
    '''
        probs is the set of prob projection, factors is the strictly k-local grammar
        probs is a dictionary e.g. (a : 0.55)
        factors is a grammar list of tuple containing each char as element e.g. [(ph,a),(sh,b),(a,c)]
    '''
    def __init__(self, probs, factors):
        self.probs = probs
        self.factors = factors

    @classmethod
    def from_file(cls, tier_file, factors_file):
        '''
            An alternative constructor that creates a pTSL object based on
            the contents of input files. The files should have the following formats:

            TIER_FILE
            <symbol 1>,<projection 1>
            <symbol 2>,<projection 2>
            ...
            <symbol n>,<projection n>

            FACTORS_FILE
            <factor 1>
            <factor 2>
            ...
            <factor n>


            <symbol i> is a single segment with no spaces. Each segment is a tier alphabet.
            <projection i> is a numeric value ranging from 0.0 to 1.0. <projection i> is
            a projection probability for <symbol i>
            <factor i> is a string of segments separated by spaces.
            Segments do not need to be single characters.

            Input:
                tier_file: A string specifying where to find the tier file
                factors_file: A string specifying where to find the factor file
        '''

        probs = {}
        with open(tier_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) > 1:
                    probs[row[0]] = float(row[1])

        factors = []
        with open(factors_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                factors.append(tuple(row[0].split()))
            if max([len(x) for x in factors]) == 1:
                warnings.warn("Spaces seem to be missing in the factor input file", SyntaxWarning)

        return pTSL(probs, factors)


    def get_kfactors(self, string):
        '''
            get k based on G. The length of the longest element is k (e.g. 2 for [(a,a),(a,b),(a,c)])
        '''

        k = max([len(x) for x in self.factors])
        # pad the string as ###string####
        padded_string = tuple((k-1) * ['#'] + string + (k-1) * ['#'])
        kfactors = []
        for i in range(0, len(padded_string) - k + 1):
            kfactors.append(padded_string[i:i + k])
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
        prob = self.probs.get(first, 0)
        projections = self.get_projections(rest)

        new_projections = []
        for proj, val in projections:
            new_projections.append(([first] + proj, prob * val))
            new_projections.append((proj, (1 - prob) * val))
        return new_projections

    def eval_conditional_probabilities(self, corpus_file,
                                       verbose=False,
                                       train=False,
                                       corpus_filename=None,
                                       projection_filename=None):
        '''
            Calculates conditional probabilities of corpus forms based on the current pTSL grammar.

            Input:
                corpus_file: path to the corpus file.
                verbose: output individual projections.
                train: do not output conditional probabilities of train.
                corpus_filename: path to save corpus conditional probabilities.
                projection_filename: path to save individual projections.
        '''
        corpus_data = self.read_corpus_file(corpus_file)
        corpus_projections = []
        for data in corpus_data:
            corpus_projections.append([data[0], self.get_projections(data[0])])
        # currently designed to create csv and print separately, needing two separate iterations of projections
        if verbose and projection_filename is not None:
            self.projections_to_csv(corpus_projections, projection_filename)

        # print
        results = []
        for index, (string, projections) in enumerate(corpus_projections):
            results.append(self.val_helper(projections))
            print("Input {}: {}".format(index + 1, ' '.join(map(str, string))))
            print("Input {} Probability: {}".format(index + 1, round(results[index], 4)))
            if verbose:
                print("\nProjection Probabilities for input {}".format(index + 1))
                for sub, value in projections:
                    if value:
                        print([' '.join(map(str, sub))], ': ', round(value, 3))
                print()

        if not train and corpus_filename is not None:
            # need corpus_data and results
            self.cond_prob_to_csv(corpus_data, results, corpus_filename)

    def projections_to_csv(self, corpus_projections, filename):
        '''
            Save the corpus projections to the output filename
        '''
        # [(['c', 'a'], 1.0), (['a'], 0.0), (['c'], 0.0), ([], 0.0)]
        if not filename:
            filename = DEFAULT_PROJECTION_FILENAME
        with open(filename, 'w', newline="") as f:
            csv_writer = csv.writer(f)
            for string, projections in corpus_projections:
                for proj, val in projections:
                    k_factors = self.get_kfactors(proj)
                    illegal_k_factors = [ele1 for ele1 in k_factors
                                         for ele2 in self.factors if ele1 == ele2]

                    csv_writer.writerow([string,
                                        ' '.join(map(str, proj)),
                                         val,
                                         str(list(map(' '.join, illegal_k_factors)))
                                         ])

    def evaluate_proj(self, proj_probs, param, corpus_probs):
        # return sum of squared errors

        for i, t in enumerate(param):
            self.probs[t] = proj_probs[i]

        sse = 0
        for string, y in corpus_probs:
            sse += (self.val(string) - y)**2

        return sse

    def train(self, corpus_file):
        '''
            Given the corpus file with corpus data and it's conditional probabilities, update the prob projections
        '''
        corpus_probs = self.read_corpus_file(corpus_file, True)
        # get list of tiers
        # from initialized ptsl (self.probs and self.factor) and corpus data, create complete list of alphabet
        param = {x for sublist in corpus_probs for x in sublist[0]}
        # remove fixed alphabets
        param = list(param - {x for x in self.probs})

        # if param is empty, meaning no free projection parameters
        if not param:
            raise ValueError('There are no free projection parameters in the tier input file')

        # create bounds
        # instead of limiting bound for fixed value, I removed it completely from the parameter
        bounds = [(0, 1) for i in range(len(param))]

        # randomly initialize parameter - this will be the input
        proj_probs = rand(len(param))
        # run the minimize function
        proj_res = minimize(self.evaluate_proj,
                            proj_probs,
                            bounds=bounds,
                            method='L-BFGS-B',
                            args=(param, corpus_probs))

    def print_optimal_projection_probabilities(self, corpus_file):
        self.train(corpus_file)
        print("Optimal projection probabilities:")
        self.print_projection_probabilities()

    def print_projection_probabilities(self):
        for key in sorted(self.probs.keys()):
            print("{}: {}".format(key, round(self.probs[key], 4)))
            print()

    def probs_to_csv(self, filename=None):
        if not filename:
            filename = DEFAULT_TIER_PROB_FILENAME
        with open(filename, 'w', newline="") as f:
            csv_writer = csv.writer(f)
            for key, value in self.probs.items():
                csv_writer.writerow([key, value])

    def pTSL_to_TSL(self, tier_filename=None, factor_filename=None):
        '''
            Convert pTSL grammar to TSL and output a TSL grammar by saving the tier and k-factor files.

            Case 1:
                If the k-factors in pTSL grammar contain only symbols with a projection prob < 1, then
                conversion is trivial.
            Case 2:
                If the k-factors in pTSL grammar contain only symbols with a projection prob of 1, then
                conversion is possible.

        '''
        if not tier_filename:
            filename = DEFAULT_TSL_TIER_FILENAME
        if not factor_filename:
            filename = DEFAULT_TSL_FACTOR_FILENAME

        # case 1 -  trivial case, all inputs accepted. Since no factor exists, choice of Tier is not relevant
        #           instead of generating two empty files, following message is printed
        if all([all([self.probs[sym] < 1 for sym in factor]) for factor in self.factors]):
            print("Trivial conversion from pTSL to TSL - accepts all inputs.")
            return

        # case 2
        if all([all([self.probs[sym] == 1 for sym in factor]) for factor in self.factors]):
            tier = [x for x in self.probs if self.probs[x] > 0]
            with open(tier_filename, 'w', newline="") as f:
                csv_writer = csv.writer(f)
                for sym in tier:
                    csv_writer.writerow(sym)

            with open(factor_filename, 'w', newline="") as f:
                csv_writer = csv.writer(f)
                for sym in self.factors:
                    csv_writer.writerow(sym)
            return

        raise ValueError("Conversion from pTSL to TSL not possible")

    @staticmethod
    def cond_prob_to_csv(corpus_data, results, filename=None):
        if not filename:
            filename = DEFAULT_PROB_FILENAME
        with open(filename, 'w', newline="") as f:
            csv_writer = csv.writer(f)
            for data, val in list(zip(corpus_data, results)):
                csv_writer.writerow([' '.join(map(str, data[0])), round(val, 6)])

    @staticmethod
    def read_corpus_file(corpus_file, train=False):
        '''
            Read corpus data from a file
            The file should have the following formats:

            <string 1>,<probability 1>
            <string 2>,<probability 1>
            ...
            <string n>,<probability 1>


            <string i> is a string of segments separated by spaces.
            Segments do not need to be single characters.

            Probabilities are optional. They are only used when train == True

            Input:
                corpus_file: A string specifying where to find the corpus file
                train: If True, save probabilities along with the corpus strings
        '''
        corpus_data = []
        with open(corpus_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            max_length = 0
            if train:
                for row in csv_reader:
                    max_length = max(max_length, len(row[0].split()))
                    try:
                        corpus_data.append([row[0].split(), float(row[1])])
                    except IndexError:
                        print("Missing corpus file probabilities")

            else:
                for row in csv_reader:
                    max_length = max(max_length, len(row[0].split()))
                    corpus_data.append([row[0].split()])

            if max_length <= 1:
                warnings.warn("Spaces seem to be missing in the corpus input file", SyntaxWarning)

        return corpus_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the probability assigned to input corpus."
    )

    parser.add_argument(
        'input_file', type=str, action='store', nargs='*',
        help='Paths to the tier-projection file, factors file, and input corpus file.'
    )

    parser.add_argument(
        '--train', action='store_true',
        help='Learn optimal projection probabilities to match input data.'
    )

    parser.add_argument(
        '--verbose', action='store_true',
        help='Prints probabilities for each possible projection.'
    )

    parser.add_argument(
        '--output_file', type=str, default=[None, None], nargs='*',
        help='Path to either tier probabilities or corpus conditional probabilities.'
             'If verbose, pass another path to projections as the second argument.'
    )

    parser.add_argument(
        '--to_tsl', action='store_true',
        help='Convert pTSL grammar to TSL grammar (if possible).'
    )

    parser.add_argument(
        '--tsl_output_file', type=str, default=[None, None], nargs=2,
        help='Path to tier file and k-factor file of TSL grammar.'
    )

    args = parser.parse_args()

    if args.to_tsl:
        input_tier = args.input_file[0]
        input_factors = args.input_file[1]
        ptsl = pTSL.from_file(input_tier, input_factors)
        ptsl.pTSL_to_TSL(args.tsl_output_file[0], args.tsl_output_file[1])

    else:
        input_tier = args.input_file[0]
        input_factors = args.input_file[1]
        input_corpus = args.input_file[2]

        if not args.output_file:
            # --output flag with no arguments
            output_file = ''
            output_proj_file = ''
        elif len(args.output_file) == 1:
            # --output flag with one argument
            output_file = args.output_file[0]
            output_proj_file = ''
        elif not args.output_file[0]:
            # no --output flag
            output_file = None
            output_proj_file = None
        else:
            # --output flag with two arguments
            output_file = args.output_file[0]
            output_proj_file = args.output_file[1]

        ptsl = pTSL.from_file(input_tier, input_factors)

        if args.train:
            # train and update ptsl
            ptsl.print_optimal_projection_probabilities(input_corpus)
            # save probs
            if output_file is not None:
                ptsl.probs_to_csv(output_file)
            # if verbose, evaluate as normal and save and print proj
            ptsl.eval_conditional_probabilities(input_corpus,
                                                args.verbose,
                                                args.train,
                                                output_file,
                                                output_proj_file)

        else:
            # evaluate (during evaluate, save and print proj if verbose)
            ptsl.eval_conditional_probabilities(input_corpus,
                                                args.verbose,
                                                args.train,
                                                output_file,
                                                output_proj_file)
