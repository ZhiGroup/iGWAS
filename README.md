# iGWAS
iGWAS is a framework for extrating phenotypes from images and doing GWAS.

## Walkthrough
The code for training segmentation network is at `segmentation`.

The code for training quality assessment network is at `quality_assessment`.

The code for training embedding network is at `embedding`.

The name of the dataset used for training segmentation network can be found in `segmentation/prepare_datasets.py`, the dataset used for training embedding network can be downloaded at [https://www.kaggle.com/c/diabetic-retinopathy-detection/data](https://www.kaggle.com/c/diabetic-retinopathy-detection/data). The dataset used for training quality assessment network was manually created using the TKinter program `quality_assessment/binary_classification_tool.py`.

The `GWAS` folder contains wrapper functions for BOLT-LMM and Plink, and some helper functions to do additional analysis.

The `locuszoom_plots` folder contains all the locuszoom plots and the script to download figures from the locuszoom website.