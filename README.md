# Investigating the power of deep learning for predicting breast cancer from whole slide images
A third year project in partial requirement for a Computer Science BSc at the
University of Warwick, supervised by [Professor Fayyaz Minhas](https://sites.google.com/view/fayyaz).

## Abstract
Both in the UK and globally, breast cancer is the most commonly diagnosed cancer in females. A key component in the diagnosis and management of breast cancer is the analysis of the tumour samples from the diagnostic biopsy.  
The computational pathology literature shows high average results for predicting tumour properties from the microscope slides using deep learning models, giving hope that the need for specific laboratory tests could be eliminated. However, recent literature shows that these models make excessive use of correlated properties and so perform poorly in subgroups of patients who subvert these correlations.  
In this project we gathered data on the relationship between the complexity of models and their overall performance. We also investigated the feasibility of making models ignore these undesired correlations.  
We find that the gap between simple models and complex models is small, suggesting that the popular status of ER/PR prediction from H\&E as a benchmark is not deserved. We also find that various cutting-edge techniques for learning robust models are each incapable of improving performance in this task over the difficult subset.  
We conclude that H\&E stains do not contain data from which ER/PR can be independently predicted, and so IHC testing cannot be directly disrupted by computational pathology.

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
- `GNNModelTrainer` must create a `LightningModule` from the torch Module that defined the architecture. This is achieved using the wrapper class [`LightningModel`](experiments/LightningModel.py); this defines the loss function and optimizer. Experiments that correspond to changes to `LightningModel` are implemented as features switches in `GNNModelTrainer` which are passed through to `LightningModel`
- [`ModelEvaluationUtils`](experiments/ModelEvaluationUtils.py) contains the definitions of our metrics
- `GNNModelTrainer` creates an instance of [`ModelEvaluator`](experiments/ModelEvaluator.py). The `evalaute_fold` method is called at the end of each fold and the `close` method is called once there are no folds left to evaluate. This produces a `.metrics` file, our own format designed to be both human-readable and machine-readable