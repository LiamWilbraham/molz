[![CircleCI](https://circleci.com/gh/LiamWilbraham/molz.svg?style=shield)](https://circleci.com/gh/LiamWilbraham/molz)
[![Code Quality Score](https://www.code-inspector.com/project/16914/score/svg)](https://www.code-inspector.com/project/16914/score/svg)
[![Code Quality Score](https://www.code-inspector.com/project/16914/status/svg)](https://www.code-inspector.com/project/16914/status/svg)

# molZ

Statistical analysis tool to help identify molecular fragments that promote, or detract from,
target properties.

Sepecifically, this tool calculates the "z-scores" of molecular substructures in a given
sub-population of a database to identify fragments that are over- or under-represented in this
sub-population relative to a reference population. These substructures can either be specified
by the user, or automatically generated using Morgan fingerprints.

## How to install

molZ relies heavily on [RDKit](https://www.rdkit.org), which I recommend installing via conda
forge:

```
$ conda install -c conda-forge rdkit
```

After that, molZ can be installed with `pip`:

```
$ pip install molz
```

## How to use

Using auto-generated fragments:

```python
from molz import ZScorer

# instantiate scorer class, optionally set length and radius of morgan fingerprint.
# In this case, data.csv is a .CSV file of two columns: SMILES and computed LogP.
scorer = ZScorer('data.csv', fp_rad=3, fp_bits=4096)

# We are going to compute zscores of fragments present in high logp molecules.
scorer.score_fragments('penalised_logp', [12, 25])

# We can plot a bar graph of zscores for the 15 highest and lowest scoring fragments.
# Also, we can draw a given fragment by refering to its Morgan fingerprint bit index.
scorer.plot(k=15, save_to='zscores_auto.png')
scorer.draw_fragment(3595)

```

Using user-defined fragments:

```python
from molz import ZScorer

# instantiate scorer class. In this case, data.csv is a .CSV file of two columns:
# SMILES and computed LogP.
scorer = ZScorer('data.csv')

# We are going to compute zscores of fragments present in high logp molecules.
scorer.score_fragments(
    'penalised_logp', [12, 25], fragment_smiles=['CCCC', 'OC', 'N(C)(C)']
)

# We can plot a bar graph of zscores for the 15 highest and lowest scoring fragments.
# Also, we can draw a given fragment by refering to its SMILES.
scorer.plot(k=15, save_to='zscores_user.png')
scorer.draw_fragment('CCCC')
```

## Z-scores : background

...
