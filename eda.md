---
title: EDA
nav_include: 1
---

# Exploratory Data Analysis









## Demographic Information

The patient demographics helped shape some of our goals and research questions. First, we noticed that all participants were at or above the age of 55. This means that our ability to make an "early" diagnosis is limited, since many of the participants got a screening done since they exhibited symptoms of cognitive impairment of some form. Second, we notice that a majority of our population is white and married. We understand that this means that our findings do not extend to a larger population.






![png](eda_files/eda_5_0.png)


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






![png](eda_files/eda_8_0.png)


### Important Predictors for Initial Diagnosis

From a histogram of baseline values for all predictors in our dataset conditional on initial diagnosis, a few predictors stood out to us, as being promising indicators of baseline diagnosis. These pertained to examination scores, which doctors heavily rely on to make their initial diagnoses. This influenced our decision to explore one such heavily influential examination, the MMSE (Mini-Mental State Examination) in particular.






![png](eda_files/eda_11_0.png)









    <seaborn.axisgrid.PairGrid at 0x1a24b417b8>




![png](eda_files/eda_12_1.png)


## Diagnoses Over Time





### MMSE scores over time






![png](eda_files/eda_16_0.png)


### Change in Diagnosis Between First Visit and Latest Visit





    (1648, 3)











![png](eda_files/eda_20_0.png)













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







## Deep Dive into MMSE



```python

```
