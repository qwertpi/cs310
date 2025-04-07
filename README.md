# Investigating the power of deep learning for predicting breast cancer from whole slide images
A third year project in partial requirement for a Computer Science BSc at the
University of Warwick, supervised by [Professor Fayyaz Minhas](https://sites.google.com/view/fayyaz).

## Abstract
TO DO

## Copyright
Copyright © 2024-25 Rory Sharp All rights reserved.

## Code structure
### datasets/
Contains code used to process the datasets. The datasets themselves must be obtained separately:
- The TCGA graphs are available on request from [Dr Mark Eastwood](mailto:Mark.Eastwood@warwick.ac.uk). The TCGA-BRCA clinical supplements used to find the ER/PR labels are publicly available for download from [NIH NCI GDC](https://portal.gdc.cancer.gov/)
- The ABCTB graphs (which already contain ER/PR labels) are also available on request from Dr Mark Eastwood. Note however that the ABCTB dataset must only be used for research purposes
### experiments/
The bulk of the code for the project:
- Each model architecture defines a subclass of `torch.nn.Module`
- 5-fold cross-validation is carried out by creating an instance of [`GNNModelTrainer`](experiments/GNNModelTrainer.py) and calling the `train_and_validate` method. The model architecture class is passed to `train_and_validate` and a new instance of it is constructed in each fold
- `GNNModelTrainer` trains the model using PyTorch Lightning's `Trainer` object with `ModelCheckpoint` and `EarlyStopping` callbacks
- `GNNModelTrainer` must create a `LightningModule` from the torch Module that defined the architecture. This is achieved using the wrapper class [`LightningModel`](experiments/LightningModel.py); this defines the loss function and optimizer
- [`ModelEvaluationUtils`](experiments/ModelEvaluationUtils.py) contains the definitions of our metrics
- `GNNModelTrainer` creates an instance of [`ModelEvaluator`](experiments/ModelEvaluator.py). The `evalaute_fold` method is called at the end of each fold and the `close` method is called once there are no folds left to evaluate. This produces a `.metrics` file, our own format designed to be both human-readable and machine-readable