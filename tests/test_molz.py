from molz import ZScorer


# def test_load_molecule_property_data():
#     scorer = ZScorer()
#     scorer._load_molecule_property_data('tests/data.csv')
#     assert 'mol' in scorer.data.columns
#     assert len(scorer.data) == 2


# def test_compute_user_frags():
#     scorer = ZScorer()
#     scorer._load_molecule_property_data('tests/data.csv')

#     scorer._compute_user_frags(['c1ccccc1', 'CCCC'])

#     assert scorer.data['c1ccccc1'].iloc[0] == 1
#     assert scorer.data['c1ccccc1'].iloc[1] == 0
#     assert scorer.data['CCCC'].iloc[0] == 0
#     assert scorer.data['CCCC'].iloc[1] == 1


# def test_compute_morgan_frags():
#     scorer = ZScorer(fp_rad=0, fp_bits=8)
#     scorer._load_molecule_property_data('tests/data.csv')

#     scorer._compute_morgan_frags()

#     assert len(scorer.data.columns) == 11  # each frag col plus originals
#     assert scorer.data[0].iloc[0] == 0
#     assert scorer.data[0].iloc[1] == 1


def test_score_fragments():
    scorer = ZScorer()
    scorer._load_molecule_property_data('tests/data.csv')

    scorer.score_fragments('molwt', [60, 100], fragment_smiles=['c1ccccc1'])
