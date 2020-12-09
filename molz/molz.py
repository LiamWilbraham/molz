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
    """Compute z-scores of molecular fragments that belong to a chosen population, with respect
    to a reference population. This score is a measure of how far these fragments lie from the
    mean of the reference population. The higher the score, the more over-represented the fragment
    is in the selected population versus the reference, the lower the score, the more under-
    represented the fragment is versus the reference.

    Normally, z-scores are computed assuming the reference population is normally distributed,
    here we treat molecules as a hypergeometric distribution, with each described using a binary
    vector. Each element of these binary vector corresponds to the presence (or absence) of a
    fragment. Fragments can either be user-defined, or auto-generated using Morgan circular
    fingerprints.
    """

    def __init__(
        self,
        data: str,
        fp_rad: int = 3,
        fp_bits: int = 4096,
        from_preprocessed_pickle: str = None,
        hide_progress: bool = False
    ) -> None:
        """Init method for ZScorer.

        Args:
            data (str): Path to a .CSV file containing molecular data. This should contain,
                at minimum, a column of molecule SMILES and a column of a computed property
                for each of these molecules.
            fp_rad (int, optional):. Morgan fingerprint radius used in auto-generated fragments.
                Defaults to 3.
            fp_bits (int, optional): Morgan fingerprint bit length used in auto-generated
                fragments. Defaults to 4096.
            from_preprocessed_pickle (str, optional): Path to pre-processed dataframe. Useful when
                dealing with large datasets Defaults to None.
            hide_progress (bool, optional): Supress progress bar outputs. Defaults to False.
        """
        # fingerprint params
        self.fp_rad = fp_rad
        self.fp_bits = fp_bits
        self.prog = hide_progress
        self.user_frags = False
        self.data = None
        self.fps = None

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
        """Compute zscores for user-defined or auto-generated fragments.

        Args:
            prop (str): Property to consider when computing zscores (from input data).
            prop_range (List[float]): Property range from which sample will be extracted to
                compute zscores.
            fragment_smiles (List[str], optional): User-defined fragments. Defaults to None,
                in which case fragments are auto-generated.
        """
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
        """Create a bar plot of top and bottom k zscoring fragments.

        Args:
            k (int, optional): Number of top and bottom scoring fragments. Defaults to 4.
            save_to (str, optional): Save plot to this path. Defaults to None.
            figsize (Tuple[int, int], optional): Plot dimensions. Defaults to None.

        Returns:
            fig: Bar plot of top and bottom k zscoring fragments.
        """

        # get top-k and bottom-k zscoring fragments
        frag_ids, frag_scores = self._get_k_min_max_zscores(k)

        # create color gradient map
        my_cmap = cm.get_cmap('RdYlGn')
        my_norm = Normalize(vmin=-max(frag_scores), vmax=max(frag_scores))

        # make plot
        figsize = (8, 4) if figsize is None else figsize
        fig, axis = plt.subplots(1, 1, figsize=figsize)
        axis.bar(frag_ids, frag_scores, color=my_cmap(my_norm(frag_scores)))
        axis.set_xlabel('z-score (std. dev.)')

        plt.xticks(rotation=90)

        if save_to:
            plt.savefig(save_to)

        return fig

    def draw_fragment(self, fragment_id: Union[str, int], show_zscore: bool = True) -> str:
        """Draw a specified fragmnet.

        Args:
            fragment_id (Union[str, int]): User-defined fragment string, or position of the
                Morgan fingerprint bit to be drawn.
            show_zscore (bool, optional): Annotate drawing with zscore. Defaults to True.

        Returns:
            str: Molecule drawing SVG.
        """

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

        bit_info = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=self.fp_rad, nBits=self.fp_bits, bitInfo=bit_info
        )

        return Draw.DrawMorganBit(
            mol,
            fragment_id,
            bit_info,
            useSVG=True,
            legend=legend
        )

    def pickle_processed_data(self, picklename: str) -> None:
        """Create a pickle file of pre-processed dataframe.

        Args:
            picklename (str): Path to pickle file.
        """
        self.data.to_pickle(picklename)

    def _get_fragment_images(self, frag_ids: List[Union[int, str]]) -> list:
        """Get an SVG image for each fragment. Not currently used.

        Args:
            frag_ids (List[Union[int, str]]): Fragments for which to produce images.

        Returns:
            list: List of fragment SVGs.
        """

        # get mol object that contains each fragment
        mols = [self._get_mol_with_frag(frag) for frag in frag_ids]

        # produce image for each fragment
        images = []
        for i, mol in enumerate(mols):
            bit_info = {}
            _ = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.fp_rad, nBits=self.fp_bits, bitInfo=bit_info
            )
            img = Draw.DrawMorganBit(mol, int(frag_ids[i]), bit_info, useSVG=True)
            images.append(img)

        return images

    def _get_mol_with_frag(self, frag_id: Union[str, int]) -> Chem.Mol:
        """Given a fragment id, return a mol containing that fragment.

        Args:
            frag_id (Union[str, int]): Fragment id.

        Returns:
            Chem.Mol: RDKit mol object of mol containing fragment.
        """
        if len(self.data[self.data[int(frag_id)] == 1]) == 0:
            return None
        return self.data[self.data[int(frag_id)] == 1].mol.iloc[0]

    def _get_k_min_max_zscores(self, k: int) -> Tuple[List, List]:
        """From all zscores, return the fragment ids and scores of the top- and bottom-k scoring.

        Args:
            k (int): Number of top- and bottom-scoring fragments to return.

        Returns:
            Tuple[List, List]: Fragment ids and scores of the top- and bottom-k scoring fragments.
        """
        frag_ids, frag_scores = [], []
        for frag, zscore in sorted(self.zscores.items(), key=lambda x: x[1]):
            frag_ids.append(str(frag))
            frag_scores.append(zscore)

        # trim to k lowest and highest zscores
        frag_ids = frag_ids[:k] + frag_ids[-k:]
        frag_scores = frag_scores[:k] + frag_scores[-k:]

        return frag_ids, frag_scores

    def _load_processed_data(self, picklename: str) -> None:
        """Load previously pickled dataframe.

        Args:
            picklename (str): Path to preprocessed data.
        """
        self.data = pd.read_pickle(picklename)

    def _load_molecule_property_data(self, datafile: str) -> None:
        """Load data from .CSV.

        Args:
            datafile (str): Path to .CSV.
        """
        self.data = pd.read_csv(datafile)
        self._compute_mols_from_smiles()

    def _compute_mols_from_smiles(self) -> None:
        """Given a list of smiles, compute the RDKit mol objects.
        """
        mols = []
        for smi in tqdm.tqdm(self.data.smiles, desc='Processing SMILES', disable=self.prog):
            mols.append(Chem.MolFromSmiles(smi))
        self.data['mol'] = mols

    def _compute_morgan_fps(self) -> None:
        """Compute a numpy array of Morgan fingerprint vectors.
        """
        fp_vects = []
        for mol in tqdm.tqdm(self.data.mol, desc='Computing fingerprints', disable=self.prog):
            fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_rad, self.fp_bits)
            array = np.zeros((0, ), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp_vect, array)
            fp_vects.append(array)

        self.fps = np.zeros((len(fp_vects), self.fp_bits))
        for i, fp_vect in enumerate(fp_vects):
            self.fps[i, :] = fp_vect

    def _compute_morgan_frags(self) -> None:
        """Place morgan fingerprints vectors into dataframe.
        """
        self._compute_morgan_fps()
        np_df = pd.DataFrame(self.fps, columns=list(range(self.fp_bits)))
        self.data = pd.concat([self.data, np_df], axis=1)

    def _compute_user_frags(self, frags: List[str]) -> None:
        """Compute presence or absence of each user-defined fragment for all molecules.

        Args:
            frags (List[str]): User-defined fragments.
        """
        frags = [(f, Chem.MolFromSmiles(f)) for f in frags]
        for smiles, mol in frags:
            self.data[smiles] = self.data.mol.apply(self._compute_user_frag_matches, args=(mol,))

    def _compute_user_frag_matches(self, mol: Chem.Mol, pattern: Chem.Mol) -> int:
        """Check if molecule contains user-defined fragment.

        Args:
            mol (Chem.Mol): Molecule considered.
            pattern (Chem.Mol): Substructure describing user-defined fragment.

        Returns:
            int: 1 if match, 0 otherwise.
        """
        if mol.HasSubstructMatch(pattern):
            return 1
        return 0

    def _compute_frag_zscore(
        self,
        frag_id: Union[str, int],
        prop: str,
        prop_range: List[float]
    ) -> float:
        """Compute zscores for a given fragment.

        Args:
            frag_id (Union[str, int]): Fragment id. Either smiles string if user defined or
                integer of morgan fingerprint bit position if auto-generated.
            prop (str): Property used to select sub-population.
            prop_range (List[float]): Property range from which sub-population is sampled.

        Returns:
            float: Fragment zscore.
        """
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
