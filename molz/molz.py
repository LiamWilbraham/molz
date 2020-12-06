import tqdm
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt


class ZScorer:

    def __init__(self, data, fp_rad=3, fp_bits=4096, from_preprocessed_pickle=False):
        # fingerprint params
        self.fp_rad = fp_rad
        self.fp_bits = fp_bits

        # zscores for fragments will be stored here
        self.zscores = {}

        if from_preprocessed_pickle:
            # load preprocessed data from pickle
            self._load_processed_data(data)
        else:
            # load data and compute rdkit mol objs
            self._load_molecule_property_data(data)

    def score_fragments(self, prop, prop_range, fragment_smiles=None):
        if fragment_smiles:
            self._compute_user_frags(fragment_smiles)
            fragments = fragment_smiles
        else:
            self._compute_morgan_frags()
            fragments = list(range(self.fp_bits))

        for frag_id in tqdm.tqdm(fragments, desc='Computing fragment z-scores'):
            self.zscores[frag_id] = self._compute_frag_zscore(
                frag_id, prop, prop_range
            )

    def plot(self):
        x, y = [], []
        for frag, zscore in self.zscores.items():
            x.append(frag)
            y.append(zscore)

        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        my_cmap = cm.get_cmap('RdYlGn')
        my_norm = Normalize(vmin=min(y), vmax=max(y))

        fix, ax = plt.subplots(1, 1)
        ax.barh(x, y, color=my_cmap(my_norm(y)))

        plt.savefig('fig.png')

    def pickle_processed_data(self, picklename):
        self.data.to_pickle(picklename)

    def _load_processed_data(self, picklename):
        self.data = pd.read_pickle(picklename)

    def _load_molecule_property_data(self, datafile):
        self.data = pd.read_csv(datafile)
        self._compute_mols_from_smiles()

    def _compute_mols_from_smiles(self):
        self.data['mol'] = self.data.smiles.apply(Chem.MolFromSmiles)

    def _compute_morgan_fps(self):
        self.fps = np.vstack([
            AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_rad, self.fp_bits)
            for mol in self.data.mol
        ])

    def _compute_morgan_frags(self):
        self._compute_morgan_fps()
        np_df = pd.DataFrame(self.fps, columns=[i for i in range(self.fp_bits)])
        self.data = pd.concat([self.data, np_df], axis=1)

    def _compute_user_frags(self, frags):
        frags = [(f, Chem.MolFromSmiles(f)) for f in frags]
        for smiles, mol in frags:
            self.data[smiles] = self.data.mol.apply(self._compute_user_frag_matches, args=(mol,))

    def _compute_user_frag_matches(self, mol, pattern):
        if mol.HasSubstructMatch(pattern):
            return 1
        return 0

    def _compute_frag_zscore(self, frag_id, prop, prop_range):
        subpop_range = (
            (self.data[prop] >= prop_range[0])
            & (self.data[prop] <= prop_range[1])
        )
        subpop = self.data[subpop_range]

        pop_total = self.data[frag_id].sum()
        selection_total = subpop[frag_id].sum()

        N = len(self.data)  # total in population
        n = len(subpop)  # total in selection
        k = pop_total  # num in population with fragment
        x = selection_total  # num in selection with fragment

        # mean and variance of hypergeometric dist.
        mean = n * k / N
        var = n * k * (N - k) * (N - n) / (N**2 * (N - 1)) + 1e-30

        # compute zscore from the above
        zscore = (x - mean) / var

        # print('N:', N)
        # print('n:', n)
        # print('k:', k)
        # print('x:', x)
        # print('mean:', mean)
        # print('var:', var)
        # print('zscore:', zscore)
        return zscore

    def _get_fragment_images_for_plotting(self, fragment_ids):
        pass
