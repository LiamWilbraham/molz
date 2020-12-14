from typing import Tuple, List, Union

import tqdm
import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors


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
    FP_TYPES = ['rdkit', 'morgan']

    def __init__(
        self,
        data: str,
        fp_rad: int = 3,
        fp_bits: int = 4096,
        fp_type: str = 'morgan',
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
            fp_type (str, optional): One of 'morgan', 'rdkit'. Defaults to 'morgan'.
            from_preprocessed_pickle (str, optional): Path to pre-processed dataframe. Useful when
                dealing with large datasets Defaults to None.
            hide_progress (bool, optional): Supress progress bar outputs. Defaults to False.
        """
        # fingerprint params
        self.fp_rad = fp_rad
        self.fp_bits = fp_bits
        self.fp_type = fp_type

        if fp_type not in self.FP_TYPES:
            raise Exception('Fingerprint type not supported.')

        self.prog = hide_progress
        self.user_frags = False
        self.data = None
        self.fps = None
        self.prop_range = None
        self.prop = None

        # zscores for fragments will be stored here
        self.zscores = {}

        if from_preprocessed_pickle:
            # load preprocessed data from pickle
            self._load_processed_data(from_preprocessed_pickle)
            self.use_preprocessed = True
        else:
            # load data and compute rdkit mol objs
            self._load_molecule_property_data(data)
            self.use_preprocessed = False

    def score_fragments(
        self,
        prop: str,
        prop_range: List[float],
        fragment_smarts: List[str] = None,
    ) -> None:
        """Compute zscores for user-defined or auto-generated fragments.

        Args:
            prop (str): Property to consider when computing zscores (from input data).
            prop_range (List[float]): Property range from which sample will be extracted to
                compute zscores.
            fragment_smarts (List[str], optional): User-defined fragments. Defaults to None,
                in which case fragments are auto-generated.
        """
        self.prop_range = prop_range
        self.prop = prop

        # user-defined fragments
        if fragment_smarts:
            self.user_frags = True
            if not self.use_preprocessed:
                self._compute_user_frags(fragment_smarts)
            fragments = fragment_smarts

        # auto-generated fragments (from morgan fp)
        else:
            if not self.use_preprocessed:
                self._generate_df_with_fragments()
            fragments = list(range(self.fp_bits))

        sample_range = (
            (self.data[prop] >= prop_range[0])
            & (self.data[prop] <= prop_range[1])
        )

        # get sample in specified property range
        sample = self.data[sample_range]

        # compute total number of times each fragment appears in data
        totals = [self.data[frag_id].sum() for frag_id in fragments]

        # compute and store fragment zscores
        i = 0
        for frag_id in tqdm.tqdm(fragments, desc='Computing fragment z-scores', disable=self.prog):
            self.zscores[frag_id] = self._compute_frag_zscore(
                frag_id, sample, totals[i]
            )
            i += 1

    def plot(
        self,
        k: int = 4,
        save_to: str = None,
        figsize: Tuple[int, int] = (8, 4),
        top_only: bool = False,
        log_y: bool = False
    ):
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
        if top_only and len(frag_ids) > 1:
            frag_ids, frag_scores = frag_ids[k:], frag_scores[k:]

        # create color gradient map
        my_cmap = cm.get_cmap('RdYlGn')
        my_norm = Normalize(vmin=-max(frag_scores), vmax=max(frag_scores))

        # make plot
        fig, axis = plt.subplots(1, 1, figsize=figsize)
        axis.bar(frag_ids, frag_scores, color=my_cmap(my_norm(frag_scores)), width=0.4, log=log_y)
        axis.set_ylabel('z-score (std. dev.)')

        plt.xticks(rotation=90)
        plt.tight_layout()

        if save_to:
            plt.savefig(save_to)

        plt.show()

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
            mol = Chem.MolFromSmarts(fragment_id)
            return Draw.MolsToGridImage(
                [mol],
                molsPerRow=1,
                subImgSize=(200, 200),
                legends=[legend]
            )

        # handle drawing of auto-generated fragments
        mol = self._get_mol_with_frag(fragment_id)

        bit_info = {}
        if self.fp_type == 'morgan':
            _ = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=self.fp_rad, nBits=self.fp_bits, bitInfo=bit_info
            )

            return Draw.DrawMorganBit(
                mol,
                fragment_id,
                bit_info,
                useSVG=True,
                legend=legend
            )

        if self.fp_type == 'rdkit':
            _ = Chem.RDKFingerprint(
                mol,
                minPath=self.fp_rad,
                maxPath=self.fp_rad,
                fpSize=self.fp_bits,
                bitInfo=bit_info,
            )

            return Draw.DrawRDKitBit(
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

    def _get_mol_with_frag(self, frag_id: Union[str, int]) -> Chem.Mol:
        """Given a fragment id, return a mol containing that fragment.

        Args:
            frag_id (Union[str, int]): Fragment id.

        Returns:
            Chem.Mol: RDKit mol object of mol containing fragment.
        """
        if self.prop_range:
            sample_range = (
                (self.data[self.prop] >= self.prop_range[0])
                & (self.data[self.prop] <= self.prop_range[1])
            )

            # get sample in specified property range
            sample = self.data[sample_range]
        else:
            sample = self.data

        # if fragment not present in range, draw mol from all data
        if len(sample[sample[int(frag_id)] == 1]) == 0:
            sample = self.data

        if len(sample[sample[int(frag_id)] == 1]) == 0:
            return None
        return sample[sample[int(frag_id)] == 1].mol.iloc[0]

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

    def _compute_fps(self) -> None:
        """Compute a numpy array of Morgan fingerprint vectors.
        """
        fp_vects = []
        for mol in tqdm.tqdm(self.data.mol, desc='Computing fingerprints', disable=self.prog):

            if self.fp_type == 'morgan':
                fp_vect = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, self.fp_rad, self.fp_bits
                )
                array = np.zeros((0, ), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp_vect, array)
                fp_vects.append(array)

            if self.fp_type == 'rdkit':
                fp_vect = Chem.RDKFingerprint(
                    mol,
                    minPath=self.fp_rad,
                    maxPath=self.fp_rad,
                    fpSize=self.fp_bits,
                )

        self.fps = np.zeros((len(fp_vects), self.fp_bits))
        for i, fp_vect in enumerate(fp_vects):
            self.fps[i, :] = fp_vect

    def _generate_df_with_fragments(self) -> None:
        """Place morgan fingerprints vectors into dataframe.
        """
        self._compute_fps()
        np_df = pd.DataFrame(self.fps, columns=list(range(self.fp_bits)))
        self.data = pd.concat([self.data, np_df], axis=1)

    def _compute_user_frags(self, frags: List[str]) -> None:
        """Compute presence or absence of each user-defined fragment for all molecules.

        Args:
            frags (List[str]): User-defined fragments.
        """
        frags = [(f, Chem.MolFromSmarts(f)) for f in frags]
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
        subpop: pd.DataFrame,
        total: int
    ) -> float:
        """Compute zscores for a given fragment.

        Args:
            frag_id (Union[str, int]): Fragment id. Either smiles string if user defined or
                integer of morgan fingerprint bit position if auto-generated.
            subpoop (DataFrame): Sample of population in specified property range.
            total (int): Total in population with fragment.

        Returns:
            float: Fragment zscore.
        """
        pop_total = total
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
