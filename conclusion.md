---
title: Conclusion
nav_include: 5
---

## Summary and Significance of Findings

Our cross-sectional models that sought to predict initial diagnosis from baseline predictors indicates that this task is fairly straightforward if given information about cognitive tests, such as the Mini-Mental State Examination and the Clinical Dementia Rating, which is intuitive since doctors and diagnosticians typically rely on these tests in order to make their diagnoses. 

Our investigation with k-Means Clustering indicates that finding a natural clustering within the data is tricky, and lends itself to the conclusion that the diagnoses given to patients may not be indicative of the bigger picture - that there are multiple levels of progression within Mild Cognitive Impairment or Alzheimer's Disease. 

Our longitudinal models, (1) predicting whether a given patient's diagnosis would either change from Cognitively Normal to either Mild Cognitive Impairment or Alzheimer's Disease and (2) whether the diagnosis would change from Mild Cognitive Impairment to Alzheimer's Disease illustrate that Gradient Boosting models have the highest AUROC and predictive power for both categories. 

Our exploration of genetic data illustrated that the gene with the largest positive coefficient in this model has no identified link with cognitive function, autophagy, protein synthesis, calcium signalling, immune system, inflammation or any other process linked with neurodegenerative disease - and is only moderately expressed in the brain, and the gene with the strongest negative coefficient has actually been previously been positively linked with amyloidogenesis in mice, contrary to what our model might imply, which suggests that there may be limited immediate benefit to using gene expression profile data. Additionally, the samples from which the data is derived were taken from blood, not the central nervous system, which means that the gene expression profiles are not indicative of the cellular environment in the brain, as the gene expression profiles of different tissues varies hugely and the central nervous system is separated from the rest of the body by a near-impervious blood-brain-barrier. 

## Limitations and Future Work

One major limitation of our dataset is that, as our EDA shows, the youngest participant in the study is aged 55. Thus, we cannot really use any of our predictors to provide "early" diagnoses. Thus, extending our methodology to datasets involving younger patients would be extremely valuable. Another limitation of our data is that a large number of participants are male, white and married, which means that our inferences may not be generalizable to a larger population. To this end, we also would like to study whether the model does particularly well for this highly represented demographic group of patients and if so, adopt an adaptive boosting method for being able to predict under-represented groups in the ADNI dataset.

Second, the nature of the classification problem we are dealing with is not simple. Alzheimer's disease cannot be measured directly, but instead must itself be "predicted" through proxies such as psychometric exams or a variety of physiological factors like amyloid-beta levels and brain volume that together signal the presence of AD. Thus, the response variable to our models is itself a prediction, and therefore introduces error into the model's accuracy that must be considered. We explored unsupervised clustering methods to identify the degree to which the dependent variable that we chose to model - diagnosis - corresponds with the natural progression of AD. However, our future work entails using other variables, or combinations of variables that match the natural progression of the disease better than the three diagnoses in the dataset. 

We might be able to identify predictors that help us move from this classification problem, which is in effect discretization error from imposing a set of three possible diagnoses to describe what is in actuality a continuous progression, to a regression problem. To measure this discretization error, it is necessary to consider the degree to which the diagnoses align with natural discontinuities (if there are any) in the progression of dementia and AD.

We also show that certain cognitive tests, such as the MMSE and the CDR, provide excellent predictive power for the initial diagnosis. However, we would like to generalize our findings here by looking at individual components from these tests and other cognitive tests such as the ADAS, such as spatial and temporal orientation or linguistic markers, to see if any particular feature correlates highly with decline in diagnosis. 
