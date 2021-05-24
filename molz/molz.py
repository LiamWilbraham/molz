from typing import Tuple, List, Union
from tabulate import tabulate

import tqdm
import numpy as np
import scipy.stats as stats

import pandas as pd
from pandasql import sqldf

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

    FP_TYPES = ["rdkit", "morgan"]

    def __init__(
        self,
        datafile: str,
        fp_rad: int = 3,
        fp_bits: int = 4096,
        fp_type: str = "morgan",
        from_preprocessed_pickle: str = None,
        hide_progress: bool = False,
        tabulate_scores: bool = True,
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
        self.datafile = datafile

        # fingerprint params
        self.fp_rad = fp_rad
        self.fp_bits = fp_bits
        self.fp_type = fp_type

        if fp_type not in self.FP_TYPES:
            raise Exception("Fingerprint type not supported.")

        self.prog = hide_progress
        self.table = tabulate_scores
        self.user_frags = False
        self.data = None
        self.fps = None
        self.props = None
        self.prop_ranges = None
        self.from_preprocessed_pickle = from_preprocessed_pickle
        self.relative_sample_size = 0
        #

        """
        Moved the processing to score_fragments, so we only import the
        necessary columns, as defined with the properties
        """
        # zscores for fragments will be stored here
        self.zscores = {}

        # load in the data on initialisation
        if self.from_preprocessed_pickle:
            # load preprocessed data from pickle
            self._load_processed_data(self.from_preprocessed_pickle)
            self.use_preprocessed = True
        else:
            # load data and compute rdkit mol objs
            self._load_molecule_property_data(self.datafile)
            self.use_preprocessed = False

        # adding Chem.Mol object to the dataframe
        self._compute_mols_from_smiles()

    def set_ranges(
        self,
        properties: List[Tuple[str, Tuple[float, float]]],
    ) -> None:
        """Define the range or ranges of properties in the data, and get a subpopulation
        of the data that meets the set criteria.

        Args:
            properties List[Tuple[str, Tuple[float, float]]]: A list of the properties and
            their respective ranges:
            [
                ('property_1'(lower_bound, upper_bound)),
                ('property_2'(lower_bound, upper_bound)),
            ]
            etc...
        """

        props = []
        prop_ranges = []
        for prop in properties:
            props.append(prop[0])
            prop_ranges.append(prop[1])
        self.prop_ranges = prop_ranges
        self.props = props

        _ = self.get_sample()

    def score_fragments(
        self,
        fragment_smarts: List[str] = None,
    ) -> None:
        """Compute zscores for user-defined or auto-generated fragments.

        Args:
            fragment_smarts (List[str], optional): User-defined fragments. Defaults to None,
                in which case fragments are auto-generated.
        """

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

        sample = self.get_sample()

        # compute total number of times each fragment appears in data
        totals = [self.data[frag_id].sum() for frag_id in fragments]

        # compute and store fragment zscores
        i = 0
        for frag_id in tqdm.tqdm(
            fragments, desc="Computing fragment z-scores", disable=self.prog
        ):
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
        log_y: bool = False,
    ) -> None:
        """Create a bar plot of top and bottom k zscoring fragments and print results to console
        as a table.

        Args:
            k (int, optional): Number of top and bottom scoring fragments. Defaults to 4.
            save_to (str, optional): Save plot to this path. Defaults to None.
            figsize (Tuple[int, int], optional): Plot dimensions. Defaults to None.

        Returns:
            fig: Bar plot of top and bottom k zscoring fragments.

        Also prints a table for the highest k zscored fragments.
        """

        # get top-k and bottom-k zscoring fragments and add to dict for tabulation
        frag_ids, frag_scores = self._get_k_min_max_zscores(k)

        if top_only and len(frag_ids) > 1:
            frag_ids, frag_scores = frag_ids[k:], frag_scores[k:]

        printable = {"Fragment": frag_ids[::-1], "z": frag_scores[::-1]}

        # create color gradient map
        my_cmap = cm.get_cmap("RdYlGn")
        my_norm = Normalize(vmin=-max(frag_scores), vmax=max(frag_scores))

        # make plot
        fig, axis = plt.subplots(1, 1, figsize=figsize)
        axis.bar(
            frag_ids,
            frag_scores,
            color=my_cmap(my_norm(frag_scores)),
            width=0.4,
            log=log_y,
        )
        axis.set_ylabel("z-score (std. dev.)")

        plt.xticks(rotation=90)
        plt.tight_layout()

        if self.table:
            print("\n" + tabulate(printable, headers="keys", tablefmt="github") + "\n")

        if save_to:
            plt.savefig(save_to)

        plt.show()

    def draw_fragment(
        self, fragment_id: Union[str, int], show_zscore: bool = True
    ) -> str:
        """Draw a specified fragmnet.

        Args:
            fragment_id (Union[str, int]): User-defined fragment string, or position of the
                Morgan fingerprint bit to be drawn.
            show_zscore (bool, optional): Annotate drawing with zscore. Defaults to True.

        Returns:
            str: Molecule drawing SVG.
        """

        # images will be annotated with zscore
        legend = f"zscore = {self.zscores[fragment_id]:.2f}" if show_zscore else ""

        # handle drawing of user-defined fragments
        if self.user_frags:
            mol = Chem.MolFromSmarts(fragment_id)
            img = Draw.MolsToGridImage(
                [mol], molsPerRow=1, subImgSize=(200, 200), legends=[legend]
            )

        # handle drawing of auto-generated fragments
        mol = self._get_mol_with_frag(fragment_id)

        bit_info = {}
        if self.fp_type == "morgan":
            _ = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=self.fp_rad, nBits=self.fp_bits, bitInfo=bit_info
            )

            img = Draw.DrawMorganBit(
                mol, fragment_id, bit_info, useSVG=True, legend=legend
            )

        if self.fp_type == "rdkit":
            _ = Chem.RDKFingerprint(
                mol,
                minPath=self.fp_rad,
                maxPath=self.fp_rad,
                fpSize=self.fp_bits,
                bitInfo=bit_info,
            )

            img = Draw.DrawRDKitBit(
                mol, fragment_id, bit_info, useSVG=True, legend=legend
            )
        return img

    def pickle_processed_data(self, picklename: str) -> None:
        """Create a pickle file of pre-processed dataframe.

        Args:
            picklename (str): Path to pickle file.
        """
        self.data.to_pickle(picklename)

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
        self.data = pd.read_csv(datafile, low_memory=True)
        self.data.insert(0, "ID", range(0, len(self.data)))

    def get_sample(self):
        """
        Querying the data and returns a sample that meets specificied criteria

        Props and prop_ranges are converted into a SQL query and executed on a temporary
        dataframe, which lacks the fingerprint bit. The sample from the temporary df is
        then joined with the main data, using the ID field in order to get the fp bits back.

        We need a temporary classless df to query from, which is deleted later. For large
        datasets, this could cause memory issues, so again, might be best to use a sqlite db
        and query that directly?
        """

        params_list = []
        tmp_df = self.data[["ID"] + list(self.props)].copy()

        for i, prop in enumerate(self.props):
            params_list.append(
                f"{prop} >= {self.prop_ranges[i][0]} AND {prop} <= {self.prop_ranges[i][1]}"
            )
        params = " AND ".join(params_list)
        sql = "SELECT * FROM tmp_df WHERE " + params + ";"

        # get sample in specified property range
        queried = sqldf(sql)
        sample = queried.merge(
            self.data,
            how="left",
            on="ID",
            suffixes=("", "__y__"),
        )
        sample.drop(
            sample.filter(regex="__y__$").columns.tolist(),
            axis=1,
            inplace=True,
        )
        del tmp_df
        self.relative_sample_size = float(len(sample) / len(self.data))
        return sample

    def _get_mol_with_frag(self, frag_id: Union[str, int]) -> Chem.Mol:
        """Given a fragment id, return a mol containing that fragment.

        Args:
            frag_id (Union[str, int]): Fragment id.

        Returns:
            Chem.Mol: RDKit mol object of mol containing fragment.
        """
        if self.prop_ranges:
            sample = self.get_sample()

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

    def _compute_mols_from_smiles(self) -> None:
        """Given a list of smiles, compute the RDKit mol objects."""
        mols = []
        for smi in tqdm.tqdm(
            self.data.smiles, desc="Processing SMILES", disable=self.prog
        ):
            mols.append(Chem.MolFromSmiles(smi))
        self.data["mol"] = mols

    def _compute_fps(self) -> None:
        """Compute a numpy array of Morgan fingerprint vectors."""
        fp_vects = []
        for mol in tqdm.tqdm(
            self.data.mol, desc="Computing fingerprints", disable=self.prog
        ):

            if self.fp_type == "morgan":
                fp_vect = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, self.fp_rad, self.fp_bits
                )

            if self.fp_type == "rdkit":
                fp_vect = Chem.RDKFingerprint(
                    mol,
                    minPath=self.fp_rad,
                    maxPath=self.fp_rad,
                    fpSize=self.fp_bits,
                )

            array = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp_vect, array)
            fp_vects.append(array)

        self.fps = np.zeros((len(fp_vects), self.fp_bits))
        for i, fp_vect in enumerate(fp_vects):
            self.fps[i, :] = fp_vect

    def _generate_df_with_fragments(self) -> None:
        """Place morgan fingerprints vectors into dataframe."""
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
            self.data[smiles] = self.data.mol.apply(
                self._compute_user_frag_matches, args=(mol,)
            )

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
        self, frag_id: Union[str, int], subpop: pd.DataFrame, total: int
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

        # Using sp just so it's easy to switch functions if need be. Granted it's a little
        # slower than previous but easy enough to switch back
        use_scipy = False
        if use_scipy:
            mean = stats.hypergeom.mean(N, n, k)
            var = stats.hypergeom.var(N, n, k) + 1e-30
        else:
            mean = n * k / N
            var = n * k * (N - k) * (N - n) / (N ** 2 * (N - 1)) + 1e-30

        return (x - mean) / var
