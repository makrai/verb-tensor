# verb-tensor
## Verb sense induction with tensor decomposition

[Paper](https://hlt.bme.hu/en/publ/vtensor22)

1. `count` triples in batches of files and `sum` them
    * debug_get_triple.ipynb
1. compute association socres with `append_pmi.py`
    * `top`.. and `weight_distri.ipynb` explore these statistics
    * `combine_freq_and_pmi.ipynb` considers combining some association scores
1. decompose the tensor with `decomp_pmi.py`
    * debugging: explore the decomposition error with `noise_redu.ipynb`
1. `eval`uate the embedding vectors in noun, verb and SVO similarity
1. analyse latent dimensions manually with `police_arrest_suspect`
1. plot more experiments with `show_exper.ipynb`
1. `cluster` embedding vectors
1. `symmetry.ipynb` is based on the hypothesis that the embedding vectors of
   subjects and objects should be the same
