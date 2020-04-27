# MOLGAN 

An implementation of the [MolGAN](https://arxiv.org/pdf/1805.11973.pdf) architecture, published by De Cao and Kip.

The MolGAN aims at generating molecules directly as graphs, without using string representations like SMILES. It produces valid molecules quite efficiently, but suffers from mode collapse - i.e. there is a severe lack of diversity in the molecules it produces. This implementation does not include the reinforcement learning part, which is supposed to push the model to output molecules with a specific desired property.

## Usage

After cloning the repo, run the script `data/download_dataset.sh` to download the dataset. Then simply run `python gan.py` and check the results in the `results` folder as they are being created.

## Libraries required

  - tensorflow >= 2.1
  - tensorflow-probability >= 0.9
  - rdkit >= 2019.09.3
  - matplotlib >= 3.1

## Citation
```
[1] De Cao, N., and Kipf, T. (2018).MolGAN: An implicit generative
model for small molecular graphs. ICML 2018 workshop on Theoretical
Foundations and Applications of Deep Generative Models.
```

BibTeX format:
```
@article{de2018molgan,
  title={{MolGAN: An implicit generative model for small
  molecular graphs}},
  author={De Cao, Nicola and Kipf, Thomas},
  journal={ICML 2018 workshop on Theoretical Foundations
  and Applications of Deep Generative Models},
  year={2018}
}

```

