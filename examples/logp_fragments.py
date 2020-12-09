from molz import ZScorer


def example_user_defined():
    scorer = ZScorer('testdata_5k.csv', fp_bits=4096, fp_rad=2)

    scorer.score_fragments(
        'penalised_logp', [12, 25], fragment_smiles=['CCCC', 'O[H]']
    )

    scorer.plot(k=15, save_to='example_penalised_logp.png')
    scorer.draw_fragment('CCCC')


def example_morgan():
    scorer = ZScorer('_testdata/testdata_5k.csv', fp_bits=4096, fp_rad=3)

    scorer.score_fragments(
        'penalised_logp', [12, 25]
    )

    scorer.plot(k=15, save_to='example_penalised_logp.png')
    scorer.draw_fragment(0)


if __name__ == '__main__':
    example_morgan()
    example_user_defined()
