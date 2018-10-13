**These scripts in Python 3.6 conform a small package to classify resting state fMRI time series from different vigilance stages by combining whole-brain modeling with machine learning tools. This package is based on the WBLEC_toolbox which has been modified and adapted.  

## fMRI-Sleep-Staging tool workflow
1. Whole-brain linear effective connectivity (WBLEC) estimation using *ParameterEstimation.py* (which calls *WBLECmodel.py*)
2. Functional connectivity measures similarity analysis using *SimilarityAnalysis.py*
3. Machinne Learning Classification using *Classification.py*


## 1. Whole-brain linear effective connectivity (WBLEC) estimation 

The script *ParameterEstimation.py* (adapted from WBLEC_toolbox) calculates the spatiotemporal functional connectivity matrices for each BOLD time series. Then, it calls the model optimization (function in *WBLECmodel.py*) and stores the model estimates (effective connectivity matrix embedded in the Jacobian J and input variances Sigma) in an array. 

Input data:
- Resting state fMRI time series for every subject, divided into four vigilance stages (awake, N1 sleep, N2 sleep, N3 sleep).
- Structural connectivity matrix (corresponding to the AAL90 parcellation).
- ROI labels (corresponding to the AAL90 parcellation).

Output:
- Effective conectivity matrices EC (embedded in the Jacobian J) 
- Spaciotemporal functional connectivity matrices (o-lag and 1-lag)
- Input variances matrix Sigma
- EC and Sigma masks 

## 2. Functional connectivity measures similarity analysis 

The script *SimilarityAnalysis.py* 



## 3. Classification

The script *Classification.py* compares the performances of two classifiers (multinomial linear regressor and 1-nearest-neighbor) in identifying subjects from EC taken as a biomarker.


## References

Data are from: Gilson M, Deco G, Friston K, Hagmann P, Mantini D, Betti V, Romani GL, Corbetta M. Effective connectivity inferred from fMRI transition dynamics during movie viewing points to a balanced reconfiguration of cortical interactions. 
Neuroimage 2017; doi.org/10.1016/j.neuroimage.2017.09.061

Model optimization is described in: Gilson M, Moreno-Bote R, Ponce-Alvarez A, Ritter P, Deco G. Estimation of Directed Effective Connectivity from fMRI Functional Connectivity Hints at Asymmetries of Cortical Connectome. PLoS Comput Biol 2016, 12: e1004762; dx.doi.org/10.1371/journal.pcbi.1004762

Classification procedure is described in: Pallares V, Insabato A, Sanjuan A, Kuehn S, Mantini D, Deco G, Gilson M. Subject- and behavior-specific signatures extracted from fMRI data using whole-brain effective connectivity. Biorxiv, doi.org/10.1101/201624

The classification script uses the scikit.learn library (http://scikit-learn.org)

