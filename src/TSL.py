import argparse
import csv
import warnings

DEFAULT_EVAL_FILENAME = "../output/evaluations.csv"
DEFAULT_PTSL_TIER_FILENAME = "../output/ptsl_tier_probs.csv"
DEFAULT_PTSL_FACTOR_FILENAME = "../output/ptsl_factors.csv"


class TSL:
    '''
        tier is the tier alphabet, k-factors is the strictly k-local grammar
        tier is a list e.g. [a, bc]
        k-factors is a grammar list of tuple containing each char as element e.g. [(ph,a),(sh,b),(a,c)]
    '''
    def __init__(self, tier, k_factors):
        self.tier = tier
        self.k_factors = k_factors
        self.k = max([len(f) for f in self.k_factors])

    @classmethod
    def from_file(cls, tier_file, factors_file):
        '''
            An alternative constructor that creates a TSL object based on
            the contents of input files. The files should have the following formats:

            TIER_FILE
            <symbol 1>
            <symbol 2>
            ...
            <symbol n>

            FACTORS_FILE
            <factor 1>
            <factor 2>
            ...
            <factor n>


            <symbol i> is a single segment with no spaces. Each segment is a tier alphabet.
            <factor i> is a string of segments separated by spaces.
            Segments do not need to be single characters.

            Input:
                tier_file: A string specifying where to find the tier file
                factors_file: A string specifying where to find the factor file
        '''

        tier = []
        with open(tier_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                tier.append(row[0])

        factors = []
        with open(factors_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                factors.append(tuple(row[0].split()))
            if max([len(x) for x in factors]) == 1:
                warnings.warn("Spaces seem to be missing in the factor input file", SyntaxWarning)

        return TSL(tier, factors)

    def project(self, string):
        '''
            Given a list of symbols as a string, return the list with all the irrelevant symbols masked out
        '''
        if not string:
            return []

        first = string[0]
        rest = string[1:]

        if first in self.tier:
            return [first] + self.project(rest)
        else:
            return self.project(rest)

    def val(self, s):
        '''
            Given a string s, return True if s is accepted by the TSL grammar, False otherwise
        '''
        projected_string = self.project(s)
        padded_string = tuple((self.k - 1) * ['#'] + projected_string + (self.k - 1) * ['#'])

        return self.check_k_factors(padded_string)

    def check_k_factors(self, s):
        '''
            Given a padded string s, return True if s is accepted by the TSL grammar, False otherwise
        '''
        if len(s) < self.k:
            return True

        k_factor = s[:self.k]
        return not (tuple(k_factor) in self.k_factors) and self.check_k_factors(s[1:])

    def eval_corpus(self, corpus_file, corpus_filename=None):
        '''
            Evaluate the corpus forms based on the current TSL grammar.

            Input:
                corpus_file: path to the corpus file.
                corpus_filename: path to save corpus evaluations.
        '''
        corpus_data = self.read_corpus_file(corpus_file)
        results = []
        for string in corpus_data:
            results.append(self.val(string))

        self.eval_to_csv(corpus_data, results, corpus_filename)

    def TSL_to_pTSL(self, tier_filename, factor_filename):
        '''
            Convert TSL grammar to pTSL and output a pTSL grammar by saving the tier and k-factor files.

        '''
        if not tier_filename:
            filename = DEFAULT_PTSL_TIER_FILENAME
        if not factor_filename:
            filename = DEFAULT_PTSL_FACTOR_FILENAME
        probs = {}

        for tier in self.tier:
            probs[tier] = 1.0

        factors = self.k_factors

        with open(tier_filename, 'w', newline="") as f:
            csv_writer = csv.writer(f)
            for key, value in probs.items():
                csv_writer.writerow([key, value])

        with open(factor_filename, 'w', newline="") as f:
            csv_writer = csv.writer(f)
            for sym in factors:
                csv_writer.writerow(sym)

    @staticmethod
    def eval_to_csv(corpus_data, results, filename=None):
        '''
            Save the result of eval to output filename
        '''
        if not filename:
            filename = DEFAULT_EVAL_FILENAME
        with open(filename, 'w', newline="") as f:
            csv_writer = csv.writer(f)
            for data, val in list(zip(corpus_data, results)):
                csv_writer.writerow([' '.join(map(str, data)), round(val, 6)])

    @staticmethod
    def read_corpus_file(corpus_file):
        '''
            Read corpus data from a file
            The file should have the following formats:

            <string 1>
            <string 2>
            ...
            <string n>


            <string i> is a string of segments separated by spaces.
            Segments do not need to be single characters.

            Input:
                corpus_file: A string specifying where to find the corpus file
        '''
        corpus_data = []
        with open(corpus_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            max_length = 0

            for row in csv_reader:
                max_length = max(max_length, len(row[0].split()))
                corpus_data.append(row[0].split())

            if max_length <= 1:
                warnings.warn("Spaces seem to be missing in the corpus input file", SyntaxWarning)

        return corpus_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_file', type=str, action='store', nargs=3,
        help='Paths to the tier file, factors file, and input corpus file.'
    )

    parser.add_argument(
        '--output_file', type=str, default=None,
        help='Path to corpus eval results.'
    )

    parser.add_argument(
        '--to_ptsl', action='store_true',
        help='Convert TSL grammar to pTSL grammar'
    )

    parser.add_argument(
        '--ptsl_output_file', type=str, default=[None, None], nargs=2,
        help='Path to tier file and k-factor file of pTSL grammar'
    )

    args = parser.parse_args()

    if args.to_ptsl:
        input_tier = args.input_file[0]
        input_factors = args.input_file[1]
        tsl = TSL.from_file(input_tier, input_factors)
        tsl.TSL_to_pTSL(args.ptsl_output_file[0], args.ptsl_output_file[1])

    else:
        input_tier = args.input_file[0]
        input_factors = args.input_file[1]
        input_corpus = args.input_file[2]
        output_file = args.output_file

        tsl = TSL.from_file(input_tier, input_factors)
        tsl.eval_corpus(input_corpus, output_file)

