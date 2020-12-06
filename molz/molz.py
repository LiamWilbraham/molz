import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


class ZScorer:

    def __init__(self, fp_rad=3, fp_bits=4096):
        self.fp_rad = fp_rad
        self.fp_bits = fp_bits

        # zscores for fragments will be stored here
        self.zscores = {}

    def score_fragments(self, prop, prop_range, fragment_smiles=None):
        if fragment_smiles:
            self._compute_user_frags(fragment_smiles)
            fragments = fragment_smiles
        else:
            self._compute_morgan_frags()
            fragments = list(range(self.fp_bits))

        for frag_id in fragments:
            self.zscores[frag_id] = self._compute_frag_zscore(
                frag_id, prop, prop_range
            )

    def plot(self):
        pass

    def _load_molecule_property_data(self, datafile):
        self.data = pd.read_csv(datafile)
        self._compute_mols_from_smiles()

    def _compute_mols_from_smiles(self):
        self.data['mol'] = self.data.smiles.apply(Chem.MolFromSmiles)

    def _compute_morgan_fps(self):
        self.data['fp'] = self.data.mol.apply(
            AllChem.GetMorganFingerprintAsBitVect, args=(
                self.fp_rad, self.fp_bits)
        )

    def _compute_morgan_frags(self):
        self._compute_morgan_fps()
        fp_array = np.vstack([arr for arr in self.data.fp])
        for i, col in enumerate(fp_array.T):
            self.data[i] = col
        self.data = self.data.drop(['fp'], axis=1)

    def _compute_user_frags(self, frags):
        frags = [(f, Chem.MolFromSmiles(f)) for f in frags]
        for smiles, mol in frags:
            self.data[smiles] = self.data.mol.apply(
                self._compute_frag_matches, args=(mol,)
            )

    def _compute_frag_matches(self, mol, pattern):
        if mol.HasSubstructMatch(pattern):
            return 1
        return 0

    def _compute_frag_zscore(self, frag_id, prop, prop_range):
        subpop_range = (
            (self.data[prop] >= prop_range[0])
            & (self.data[prop] <= prop_range[1])
        )
        subpop = self.data[subpop_range]

        print('')
        print(self.data)
        print(subpop)

        pop_total = self.data[frag_id].sum()
        subpop_total = subpop[frag_id].sum()

        print(pop_total)
        print(subpop_total)

        N = len(self.data)
        k = pop_total
        n = len(subpop)
        x = subpop_total
        mean = n * k / N
        var = n * k * (N - k) * (N - n) / (N**2 * (N - 1))

        zscore = (x - mean) / var
        print(zscore)

        return zscore
