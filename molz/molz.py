from typing import Tuple, List, Union

import tqdm
import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs


class ZScorer:

    def __init__(
        self,
        data: str,
        fp_rad: int = 3,
        fp_bits: int = 4096,
        from_preprocessed_pickle: str = None,
        hide_progress: bool = False
    ) -> None:
        # fingerprint params
        self.fp_rad = fp_rad
        self.fp_bits = fp_bits
        self.prog = hide_progress
        self.user_frags = False

        # zscores for fragments will be stored here
        self.zscores = {}

        if from_preprocessed_pickle:
            # load preprocessed data from pickle
            self._load_processed_data(data)
        else:
            # load data and compute rdkit mol objs
            self._load_molecule_property_data(data)

    def score_fragments(
        self,
        prop: str,
        prop_range: List[float],
        fragment_smiles: List[str] = None
    ) -> None:

        # user-defined fragments
        if fragment_smiles:
            self.user_frags = True
            self._compute_user_frags(fragment_smiles)
            fragments = fragment_smiles

        # auto-generated fragments (from morgan fp)
        else:
            self._compute_morgan_frags()
            fragments = list(range(self.fp_bits))

        # compute and store fragment zscores
        for frag_id in tqdm.tqdm(fragments, desc='Computing fragment z-scores', disable=self.prog):
            self.zscores[frag_id] = self._compute_frag_zscore(
                frag_id, prop, prop_range
            )

    def plot(self, k: int = 4, save_to: str = None, figsize: Tuple[int, int] = None):

        # get top-k and bottom-k zscoring fragments
        x, y = self._get_k_min_max_zscores(k)

        # create color gradient map
        my_cmap = cm.get_cmap('RdYlGn')
        my_norm = Normalize(vmin=-max(y), vmax=max(y))

        # make plot
        figsize = (8, 4) if figsize is None else figsize
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.bar(x, y, color=my_cmap(my_norm(y)))
        ax.set_xlabel('z-score (std. dev.)')

        plt.xticks(rotation=90)

        if save_to:
            plt.savefig(save_to)

        return fig

    def draw_fragment(self, fragment_id: Union[str, int], show_zscore: bool = True) -> str:

        # images will be annotated with zscore
        legend = f'zscore = {self.zscores[fragment_id]:.2f}' if show_zscore else ''

        # handle drawing of user-defined fragments
        if self.user_frags:
            mol = Chem.MolFromSmiles(fragment_id)
            return Draw.MolsToGridImage(
                [mol],
                molsPerRow=1,
                subImgSize=(200, 200),
                legends=[legend]
            )

        # handle drawing of auto-generated fragments
        mol = self._get_mol_with_frag(fragment_id)

        bi = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=self.fp_rad, nBits=self.fp_bits, bitInfo=bi
        )

        return Draw.DrawMorganBit(
            mol,
            fragment_id,
            bi,
            useSVG=True,
            legend=legend
        )

    def pickle_processed_data(self, picklename: str) -> None:
        self.data.to_pickle(picklename)

    def _get_fragment_images(self, frag_ids: List[Union[int, str]]) -> list:

        # get mol object that contains each fragment
        mols = [self._get_mol_with_frag(frag) for frag in frag_ids]

        # produce image for each fragment
        images = []
        for i, m in enumerate(mols):
            bi = {}
            _ = AllChem.GetMorganFingerprintAsBitVect(
                m, radius=self.fp_rad, nBits=self.fp_bits, bitInfo=bi
            )
            img = Draw.DrawMorganBit(m, int(frag_ids[i]), bi, useSVG=True)
            images.append(img)

        return images

    def _get_mol_with_frag(self, frag_id: Union[str, int]) -> Chem.Mol:
        if len(self.data[self.data[int(frag_id)] == 1]) == 0:
            return None
        return self.data[self.data[int(frag_id)] == 1].mol.iloc[0]

    def _get_k_min_max_zscores(self, k: int) -> Tuple[List, List]:
        x, y = [], []
        for frag, zscore in sorted(self.zscores.items(), key=lambda x: x[1]):
            x.append(str(frag))
            y.append(zscore)

        # trim to k lowest and highest zscores
        x = x[:k] + x[-k:]
        y = y[:k] + y[-k:]

        return x, y

    def _load_processed_data(self, picklename: str) -> None:
        self.data = pd.read_pickle(picklename)

    def _load_molecule_property_data(self, datafile: str) -> None:
        self.data = pd.read_csv(datafile)
        self._compute_mols_from_smiles()

    def _compute_mols_from_smiles(self) -> None:
        mols = []
        for smi in tqdm.tqdm(self.data.smiles, desc='Processing SMILES', disable=self.prog):
            mols.append(Chem.MolFromSmiles(smi))
        self.data['mol'] = mols

    def _compute_morgan_fps(self) -> None:
        fps = []
        for mol in tqdm.tqdm(self.data.mol, desc='Computing fingerprints', disable=self.prog):
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_rad, self.fp_bits)
            array = np.zeros((0, ), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, array)
            fps.append(array)

        self.fps = np.zeros((len(fps), self.fp_bits))
        for i, fp in enumerate(fps):
            self.fps[i, :] = fp

    def _compute_morgan_frags(self) -> None:
        self._compute_morgan_fps()
        np_df = pd.DataFrame(self.fps, columns=[i for i in range(self.fp_bits)])
        self.data = pd.concat([self.data, np_df], axis=1)

    def _compute_user_frags(self, frags) -> None:
        frags = [(f, Chem.MolFromSmiles(f)) for f in frags]
        for smiles, mol in frags:
            self.data[smiles] = self.data.mol.apply(self._compute_user_frag_matches, args=(mol,))

    def _compute_user_frag_matches(self, mol: Chem.Mol, pattern: Chem.Mol) -> int:
        if mol.HasSubstructMatch(pattern):
            return 1
        return 0

    def _compute_frag_zscore(
        self,
        frag_id: Union[str, int],
        prop: str,
        prop_range: List[float]
    ) -> float:
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
        return (x - mean) / var
