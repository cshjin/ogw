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
  <!-- `pip install qpsolvers nsopy` -->

* install the package as develop mode
  `python setup.py develop`

## dataset

* download [MNIST 2D](https://www.kaggle.com/cristiangarcia/pointcloudmnist2d) from Kaggle.
* TUDataset benchmark will be loaded from `pyg`.

## Experiments

We provide a set of demonstrations of `OGW`:

* `tightness_syn.ipynb`: A demo of tightness on synthetic data.
* `tightness_mutag.ipynb`: A demo of tightness on MUTAG dataset.
* `barycenter_syn.ipynb`: A demo of barycenter on synthetic data.
* `barycenter_mnist_2d.ipynb`: A demo of barycenter on point cloud MNIST-2D data.

## License

The project is under [MIT](./LICENSE) license.