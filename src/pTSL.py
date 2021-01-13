class pTSL():
    def __init__(self, probs, factors):
        self.probs = probs
        self.factors = factors

    def get_kfactors(self, string):
        k = max([len(x) for x in self.factors])
        padded_string = tuple((k-1) * ['#'] + string + (k-1) * ['#'])
        kfactors = []
        for i in range(0, len(padded_string) - k + 1):
            kfactors.append(padded_string[i:i+k])
        return kfactors

    def val(self, string):
        result = self.val_helper(string)
        total_prob = 0
        for proj, val in result:
            k_factors = self.get_kfactors(proj)
            if not any([x in k_factors for x in self.factors]):
                total_prob += val
        return total_prob

    def val_helper(self, string):
        if not string:
            return [([], 1)]

        first, *rest = string
        prob = self.probs[first]
        projections = self.val_helper(rest)

        new_projections = []
        for proj, val in projections:
            new_projections.append(([first] + proj, prob * val))
            new_projections.append((proj, (1 - prob) * val))

        return new_projections
