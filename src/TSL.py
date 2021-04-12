import csv
import warnings

class TSL():
    def __init__(self, tier, k_factors):
        self.tier = tier
        self.k_factors = k_factors
        self.k = max([len(f) for f in self.k_factors])

    @classmethod
    def from_file(cls, tier_file, factors_file):
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
        if not string:
            return []

        first = string[0]
        rest = string[1:]

        if first in self.tier:
            return first + self.project(rest)
        else:
            return self.project(rest)

    def val(self, s):
        projected_string = self.project(s)
        padded_string = tuple((self.k - 1) * ['#'] + projected_string + (self.k - 1) * ['#'])

        return self.check_k_factors(padded_string)

    def check_k_factors(self, s):
        if len(s) < self.k:
            return True

        k_factor = s[:self.k]
        return not (tuple(k_factor) in self.k_factors) and self.check_k_factors(s[1:])

