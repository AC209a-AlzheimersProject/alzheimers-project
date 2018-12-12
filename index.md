---
title: Introduction
---

### Team 41: Nathan Einstein, Rory Maizels, Robbert Struyven, Abhimanyu Vasishth
AC209a Fall 2018

## Problem Description

Here is the problem description. More.

## Motivation

Here is the motivation.

## Data Source Description

The ADNI study is made up of four distinct phases -- ADNI1, ADNIGO, ADNI2, and ADNI3 -- in each of which new participants were recruited while existing participants from earlier phases continued to be monitored. This structure lends itself well to longitudinal modeling, though the variability in the information gathered on patients and inconsistent coding of variables across cohorts presents numerous data-cleaning and modeling challenges. 

Numerous relational tables are made available with detailed information on patient background, clinical evaluations, cognitive test results, genetic information, PET and MRI images, and biomarker measurements, though the information in these tables are often inconsistent between study phases or missing for certain groups of patients, follow-up intervals, or phases. We therefore drew primarily on a prepared dataset (“ADNImerge”) containing key variables for patients across all four study phases. We also explored in greater detail original datasets on gene expression and family history -- two topics that we viewed as of potentially great importance in making predictions of change in patient outcomes.

## Research Questions

Our EDA revealed that cognitive assessment scores could account for much of the variability in patient diagnoses, in large part given the heavy reliance on these exams for the definition of the ADNI diagnosis classes. A model designed to predict diagnosis involving these exams as predictors therefore seemed of relatively little value. Thus, we instead plan to predict a patient’s prognosis (i.e. change over time) given the information available at baseline (t=0). We will therefore develop two separate models for individuals who begin as cognitively normal, versus those diagnosed at baseline as MCI. In addition to diagnosis, we are also considering the option of predicting change in Mini-Mental State Examination (MMSE) score -- a 30-question exam widely-used to measure cognitive impairment, and which plays a central role in classifying new ADNI patients (indeed, it is one of the strongest-correlated predictors with diagnosis). The MMSE scores are more variable than diagnoses, especially among MCI patients, as depicted in the following plots of MMSE scores vs time after baseline for patients with different initial diagnoses.

Secondly, the clear importance of exam scores necessitates a deeper dive into the various exams -- in particular, the MMSE, CDRSB, and ADAS -- and the specific aspects of dementia that they measure. While neurological exams are clearly the most cost-effective means of identifying Alzheimer’s, it remains to be seen how well these exam scores, in combination with other factors, can predict change in diagnosis.

For the AC209 component of this project, we use several unsupervised learning algorithms to identify clusters of common groups along the AD/dementia spectrum that mimic the AD/LMCI/EMCI/CN diagnoses given to patients in the ADNI studies. 

## Related Work

Here is our related work.
