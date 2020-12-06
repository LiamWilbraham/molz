import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


class ZScorer:

    def __init__(self):
        self.fp_rad = 3
        self.fp_bits = 4096
        self.zscores = {}

    def score_fragments(self):
        pass

    def plot(self):
        pass

    def _load_molecule_property_data(self, datafile):
        self.data = pd.read_csv(datafile)

    def _compute_molsfrom_smiles(self):
        self.data['mol'] = self.data.smiles.apply(Chem.MolFromSmiles)

    def _compute_morgan_fps(self, mol):
        self.data['fp'] = self.data.mol.apply(
            AllChem.GetMorganFingerprintAsBitVect, args=(self.fp_rad, nBits=self.fp_bits)
        )

    def _compute_morgan_frags(self):
        pass

    def _compute_user_frags(self):
        pass

    def _compute_frag_zscore(self):
        pass

    def _compute_all_frag_zscores(self):
        pass
