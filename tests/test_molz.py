from pytest import approx
from molz import ZScorer


def test_load_molecule_property_data():
    scorer = ZScorer('tests/data.csv')
    assert 'mol' in scorer.data.columns
    assert len(scorer.data) == 2


def test_compute_user_frags():
    scorer = ZScorer('tests/data.csv')
    scorer._compute_user_frags(['c1ccccc1', 'CCCC'])

    assert scorer.data['c1ccccc1'].iloc[0] == 1
    assert scorer.data['c1ccccc1'].iloc[1] == 0
    assert scorer.data['CCCC'].iloc[0] == 0
    assert scorer.data['CCCC'].iloc[1] == 1


def test_compute_morgan_frags():
    scorer = ZScorer('tests/data.csv', fp_rad=0, fp_bits=8)
    scorer._compute_morgan_frags()

    assert len(scorer.data.columns) == 11  # each frag col plus originals
    assert scorer.data[0].iloc[0] == 0
    assert scorer.data[0].iloc[1] == 1


def test_score_fragments_user_defined():
    scorer = ZScorer('tests/data.csv')

    scorer.score_fragments(
        'molwt', [60, 100], fragment_smiles=['c1ccccc1', 'O', 'CCC']
    )

    assert scorer.zscores['c1ccccc1'] == approx(2.0)
    assert scorer.zscores['O'] == approx(0.0)
    assert scorer.zscores['CCC'] == approx(-2.0)
    # print(scorer.zscores)


def test_score_fragments_morgan():
    scorer = ZScorer('tests/data.csv', fp_bits=8, fp_rad=0)

    scorer.score_fragments(
        'molwt', [60, 100]
    )

    assert scorer.zscores[0] == approx(-2.0)
    for i in range(1, 8):
        assert scorer.zscores[i] == approx(0.0)


def test_pickle_processed_data():
    scorer = ZScorer('tests/data.csv')

    scorer.score_fragments(
        'molwt', [60, 100], fragment_smiles=['c1ccccc1', 'O', 'CCC']
    )

    scorer.pickle_processed_data('pytestpickle.pkl')

    # set data attrib to none to ensure load works as expected
    scorer.data = None

    scorer._load_processed_data('pytestpickle.pkl')

    assert len(scorer.data) == 2


def test_plot():
    scorer = ZScorer('tests/data.csv')

    scorer.score_fragments(
        'molwt', [60, 100], fragment_smiles=['c1ccccc1', 'O', 'CCC']
    )

    scorer.plot()


test_plot()
