---
title: Introduction
---

### Team 41: Nathan Einstein, Rory Maizels, Robbert Struyven, Abhimanyu Vasishth
AC209a Fall 2018 

## Problem Description and Motivation

Alzheimer’s disease (AD) incurs a significant toll not just on the elderly individuals who are most prone to the disease, but to their caregivers and the population at large. As the sixth leading cause of death among American adults, the disease currently affects 1.6% of the American population, though the figure is projected to double to 3.3% by 2060 as the nation’s population ages and lifespans increase.[1]

While numerous risk factors and precursors for AD have been identified in the medical literature, diagnosis of this degenerative disease in practice remains difficult, with high rates of diagnostic failure and misdiagnosis (i.e. Type I and II error) especially at early onset. [2] We therefore seek to gain insight into which factors already known to reveal the presence of or susceptibility to AD might be used, individually or in combination, to identify likelihood of change in diagnosis over time, both among those already diagnosed and those who are cognitively normal.

## Data Source Description

The ADNI study is made up of four distinct phases -- ADNI1, ADNIGO, ADNI2, and ADNI3 -- in each of which new participants were recruited while existing participants from earlier phases continued to be monitored. This structure lends itself well to longitudinal modeling, though the variability in the information gathered on patients and inconsistent coding of variables across cohorts presents numerous data-cleaning and modeling challenges.

Numerous relational tables are made available with detailed information on patient background, clinical evaluations, cognitive test results, genetic information, PET and MRI images, and biomarker measurements, though the information in these tables are often inconsistent between study phases or missing for certain groups of patients, follow-up intervals, or phases. We therefore drew primarily on a prepared dataset (“ADNImerge”) containing key variables for patients across all four study phases. We also explored in greater detail original datasets on gene expression and family history -- two topics that we viewed as of potentially great importance in making predictions of change in patient outcomes.

## Research Questions

Early exploratory data analysis revealed that cognitive assessment scores could account for much of the variability in patient diagnoses, in large part given the heavy reliance on these exams for the definition of the ADNI diagnosis classes. A model designed to predict diagnosis involving these exams as predictors therefore seemed of relatively little value. Thus, we instead focus on identifying strong predictors of a patient’s prognosis (i.e. evolution over time) given the information available at the patient's baseline visit (t=0). We therefore develop two separate models: one for individuals who begin as cognitively normal, one for those diagnosed at baseline as having mild cognitive impairment.

Secondly, the clear importance of exam scores necessitates a deeper dive into the various exams -- in particular, the MMSE, CDRSB, and ADAS -- and the specific aspects of dementia that they measure. While neurological exams are clearly the most cost-effective means of identifying Alzheimer’s, it remains to be seen how well these exam scores, in combination with other factors, can predict change in diagnosis.

For the AC209 component of this project, we use several unsupervised learning algorithms to identify clusters of common groups along the AD/dementia spectrum that mimic the AD/LMCI/EMCI/CN diagnoses given to patients in the ADNI studies, to better understand to what degree these imposed diagnoses reflect natural breaks in the evolution of dementia.

## Conclusions



## Future Directions






## Sources:

[1] Matthews, Kevin A. et al. "Racial and ethnic estimates of Alzheimer's disease and related dementias in the United States (2015–2060) in adults aged ≥65 years." Alzheimer's & Dementia: The Journal of the Alzheimer's Association, Volume 0 , Issue 0.

[2] Small, Gary W. "Early diagnosis of Alzheimer's disease: update on combining genetic and brain-imaging measures", Dialogues Clin Neurosci. 2000 Sep; 2(3): 241–246.
