## Overview
An implementation for calculating Mixed Graphical models based on the approach by Lee & Hastie (2012). Additionally, prior group knowledge about features can be added to increase model performance.

For an R version of the algorithm, see: https://github.com/Roko4/PriOmics

## Usage

### Standard MGM
Employing the standard MGM algorithm (without PriOmics extension) requires at minimum 3 inputs:
- X:	Continuous data a snumpy.ndarray with samples in rows and features in columns (n x p)
- Y:	Discrete/categorical data as numpy.ndarray with samples in rows and features in columns (n x p)
- lambda_seq: numpy.ndarray of tuning parameter lambda used for model selection. 

### PriOmics MGM
In addition to the inputs of the standard MGM, we need 2 more arguments:
- groups_X: List of groups for prior assumption (for continuous data)
- prior_X: List of the prior assumptions, either "A" or "B". If no prior should be assigned to set the corresponding feature to a group size of one. The prior will then be inored. Applying both priors to different feature groups in the same model is possible.

## Citation
Kosch, R., Limm, K., Staiger, A. M., Kurz, N. S., Seifert, N., Oláh, B., ... & Altenbuchinger, M. (2023). PriOmics: integration of high-throughput proteomic data with complementary omics layers using mixed graphical modeling with group priors. bioRxiv, 2023-11.

## Additional Resources
Altenbuchinger, M., Weihs, A., Quackenbush, J., Grabe, H. J., & Zacharias, H. U. (2020). Gaussian and Mixed Graphical Models as (multi-) omics data analysis tools. Biochimica et Biophysica Acta (BBA)-Gene Regulatory Mechanisms, 1863(6), 194418.

Altenbuchinger, M., Zacharias, H. U., Solbrig, S., Schäfer, A., Büyüközkan, M., Schultheiß, U. T., ... & Gronwald, W. (2019). A multi-source data integration approach reveals novel associations between metabolites and renal outcomes in the German Chronic Kidney Disease study. Scientific reports, 9(1), 13954.
