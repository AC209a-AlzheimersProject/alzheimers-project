---
title: Exploratory Data Analysis
notebook: eda.ipynb
nav_include: 1
---

## Contents
{:.no_toc}
*  
{: toc}










## Description of Data

### Description of ADNI Merge

The ADNI study is made up of four distinct phases – ADNI1, ADNIGO, ADNI2, and ADNI3 – in each of which new participants were recruited while existing participants from earlier phases continued to be monitored. Our main dataset is the ADNI Merge dataset, from the [Alzheimer's Disease Neuroimaging Initiative](http://adni.loni.usc.edu/).  This is essentially a dataset combining key predictors from all four phases, assembled using various sources of data within the ADNI repository. A high-level overview of the categories is as follows:

1. **Cognitive Tests**: some of the predictors, such as MMSE (Mini-Mental State Examination), CDRSB (Clinical Dementia Rating Sum of Boxes), ADAS (Alzheimer’s Disease Assessment Scale–Cognitive subscale) and RAVLT (Rey Auditory Verbal Learning Test) come from cognitive tests that clinicians used to base the patient diagnoses on.
2. **Demographic Variables**: such as Age, Gender, Marriage status and Education levels.
4. **Brain-Related Variables**: these variables, such as Hippocampus, Ventricles, WholeBrain, Entorhinal, Fusiform and MidTemp) measure various aspects of the brain.
3. **Important Biomarkers**: biomarkers, such as A-beta (Amyloid Beta), Tau, APOE4 and FDG, are important proteins or biomarkers that are associated with Alzheimer’s disease or Mild Cognitive Impairment in a lot of medical literature about the disease.

The table below provides a more detailed overview of the key variables present in our dataset, the range of values that these variables take on and the percentage of missing values, which is a key component to take into consideration with this dataset.

![png](eda_files/tests.png "Tests")

![png](eda_files/medical.png "Medical")

### Missing Values

Given the strict data-collection protocol followed and strict criteria for selecting patients so as to prevent drop-out, we expect that missing values are typically Missing At Random (MAR); missingness is due solely to observable factors such as the follow-up time (i.e. not all variables are re-collected at every follow-up), the patient’s initial diagnosis, and the particular ADNI phase (given that the data are currently in longform). Especially where the missingness is due to differing procedures carried out for patients with different baseline diagnoses, we must ensure that the way we handle missing values does not introduce bias in our model.






![png](eda_files/eda_9_0.png)


The plot above shows that certain categories, such as the Everyday Cognition (Ecog) are missing in general for the ADNI 1 phase, whereas the brain related predictors are missing for ADNI 3. One important observation is that the important biolarkers (such as Amyloid Beta, Tau and PTau) are missing partially throughout the data. The reason for this is that these are taken from a patient's Cerebraospinal Fluid ([CSF](https://medlineplus.gov/ency/article/003428.htm)), which requires an invasive procedure to obtain. Therefore, not all patients undergo this procedure during a baseline test and we see a lot of missing values here.

## Demographic Information

The patient demographics helped shape some of our goals and research questions. First, we noticed that all participants were at or above the age of 55. This means that our ability to make an "early" diagnosis is limited, since many of the participants got a screening done since they exhibited symptoms of cognitive impairment of some form. Second, we notice that a majority of our population is white and married. We understand that this means that our findings do not extend to a larger population.






![png](eda_files/eda_13_0.png)


## Initial Diagnoses

The baseline diagnoses are encoded in two ways in the ADNI Merge Dataset. The first encoding is as follows:

1. CN: Cognitively Normal
2. MCI: Mild Cognitive Impairment
3. Dementia: Alzheimer's Disease or other Dementia.

The second encoding is as follows:

1. CN: Cognitively Normal
2. EMCI: Early Mild Cognitive Impairment
3. LMCI: Late Mild Cognitive Impairment
4. SMC: Significant Memory Concerns
5. AD: Alzheimer's Disease

We choose the first encoding, changing 'Dementia' to 'AD' since there is an equivalency in the encoding of these categories. The second encoding is present only in the baseline diagnoses and not in the subsequent diagnoses, and part of our modeling is to predict future decline, so we choose the first encoding to remain consistent between diagnoses at different points of time.






![png](eda_files/eda_16_0.png)


### Important Predictors for Initial Diagnosis

From a histogram of baseline values for all predictors in our dataset conditional on initial diagnosis, a few predictors stood out to us, as being promising indicators of baseline diagnosis. These pertained to examination scores, which doctors heavily rely on to make their initial diagnoses. This influenced our decision to explore one such heavily influential examination, the MMSE (Mini-Mental State Examination) in particular.

The histograms below correspond to four cognitive tests that show a promising separability between the three classes at baseline levels. We further explore these four cognitive tests and three tests from the ADAS (Alzheimer's Disease Assessment Scale) test in a pair plot that also shows promise in a combination of cognitive tests being used to predict initial diagnoses.






![png](eda_files/eda_19_0.png)









    <seaborn.axisgrid.PairGrid at 0x1a20987e48>




![png](eda_files/eda_20_1.png)


## Diagnoses Over Time

In addition to being able to predict initial diagnosis, we also aim to predict future decline i.e. patients who were initially diagnosed as Cognitively Normal and later were diagnosed with Mild Cognitive Impairment, or patients who were initially diagnosed with Mild Cognitive Impairment and later became diagnosed with Alzheimer's Disease. In order to do this, we have a few different visuals to get a sense of the distribution of diagnoses over time and how MMSE scores (an important determiner of diagnosis) vary over time for patients.






![png](eda_files/eda_23_0.png)


We observe that the total number of diagnoses of each kind drops over time, which may perhaps be a result of patients not carrying out a follow up study for a number of different reasons.

### MMSE scores over time

Next, we show how MMSE scores vary over time for participants. We plot trajectories of MMSE scores for each of the three diagnosis types. We anticipate there to be a gradual decline in MMSE scores for the AD and MCI categories and perhaps a smaller decline for the CN category but this plot helps discern between the different groups based on rate of dropoff of MMSE scores.






![png](eda_files/eda_27_0.png)


As hypothesized, the dropoff of MMSE scores for the MCI group is faster and larger than that for the CN group. It is striking to observe that many patients initially diagnosed with AD have no observations after 3 years.

### Change in Diagnosis Between First Visit and Most Recent Visit

Using our plots above, we turn to the question of predicting change in diagnoses between first and most recent visit for a particular patient. We subset our dataset to include only patients who have multiple observations and show the number of diagnoses of each kind at baseline and at patients' most recent visit. One extremely important caveat is that the most recent visit can be 2 years later or 12 years later, and the time between first and last visit is an extremely important predictor that we keep in mind when modeling.





    (1648, 3)











![png](eda_files/eda_33_0.png)


A general pattern seems to be that the total number of MCI and CN diagnoses has decreased while the number of AD diagnoses has increased over time, which is unsurprising.

We create a table, with each entry representing a participant's initial and most recent diagnoses. A majority of participants start and end with the same diagnoses, however a subset of participants do decline, especially from MCI to AD. A non-trivial number of participants move from MCI to CN, which is worth noting. Of the 1648 participants with multiple observations, only 204 decline, which is about **12.3%** of our total sample. This is an important finding that suggests that we should perform classification with balanced class weights in our modeling phase.

The Sankey Diagram below illustrates changes in diagnoses and highlights some major trends: that a majority of patients have the same initial and most recent diagnoses.








<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DX</th>
      <th>End CN</th>
      <th>End MCI</th>
      <th>End AD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Start CN</td>
      <td>555.0</td>
      <td>48.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Start MCI</td>
      <td>39.0</td>
      <td>578.0</td>
      <td>156.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Start AD</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>259.0</td>
    </tr>
  </tbody>
</table>
</div>



![png](eda_files/sankey.png "Sankey")
