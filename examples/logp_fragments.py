from molz import ZScorer

def example():
    scorer = ZScorer('testdata_5k.csv', fp_bits=4096, fp_rad=3)

    scorer.score_fragments(
        'penalised_logp', [15, 30]
    )

    scorer.plot(k=15, save_to='example_penalised_logp.png')

    x, y = [], []
    k = 15
    for frag, zscore in sorted(scorer.zscores.items(), key=lambda x: x[1]):
        x.append(frag)
        y.append(zscore)
        x = x[:k] + x[-k:]
        y = y[:k] + y[-k:]

    for i, j in zip(x, y):
        print(i, j)


if __name__ == '__main__':
    example()
