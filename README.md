# Orthogonal Gromov-Wasserstein Distance

## Preparation

Run the automatic env setup file `source setup.sh` or 

* create virtual env (restrict to python=3.7, which is compatible with CPLEX 12.10)
  `conda create -n ogw python=3.7`

* install pytorch / pytorch-geometric
  `conda install pytorch=1.8.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`
  `pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html`
  `pip install torch-geometric pot`

* install supplimentary libraries
  `conda install matplotlib joblib networkx numba`
  `pip install qpsolvers nsopy`

* install the package as develop mode
  `python setup.py develop`

## dataset

* download [MNIST 2D](https://www.kaggle.com/cristiangarcia/pointcloudmnist2d) from Kaggle.
* TUDataset benchmark will be loaded from `pyg`.

## Experiments

We provided a comprehensive notebook `demo.ipynb` to show the idea of

* tractable bounds of FGW
* convex extension of FGW
* certification and attack for task of graph classification

Other files includes:

* `demo_train.py`: build the model for certificate and attack.
* `demo_spla.py`: preprocess to generate the linear mapping matrix $\mathcal{A}$.
* `demo_certify.py`: complete setup for experiments of robust certifications.
* `demo_attack.py`: complete setup for experiments of attack.

## License

The project is under [MIT](./LICENSE) license.