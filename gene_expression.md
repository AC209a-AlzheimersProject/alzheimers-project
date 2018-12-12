---
title: Gene Expression Profiles
notebook: gene_expression.ipynb
nav_include: 4
---

## Contents
{:.no_toc}
*  
{: toc}

## Gene Expression Profiles



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sns.set()
```


## Gene Expression Profiles
----
Consideration of an Alternative Question: Genetic Predictors
Rather than creating a predictive model of diagnosis using various types of patient information, we had previously considered developing a model from the detailed biological data available to consider the importance of genes on chromosome 21 in Alzheimer’s Disease. Chromosome 21 is known to be very important in the etiology of AD - for example, over half of those born with an extra copy of Chr21 (a condition known as Down’s Syndrome) will go on to develop Alzheimer’s Disease [1]. While it is commonly said that the link between Chr21 and AD is due to the gene for the Amyloid Precursor Protein (APP) [2], there are many other genes on this chromosome that have also been linked to Alzheimer’s disease [3].
Of the wealth of genetic data that ADNI publishes, we drew primarily on the microarray gene expression profile dataset, which used ~50,000 genetic probes to assess the activity of genes across the genome. The outcome for each patient in the gene profile dataset was determined using the ADNIMerge dataset. To identify only chromosome 21 genes, the Affymetrix gene annotation dataset was used to annotate the gene expression data set with chromosomal location of the target gene for every probe. The combination of these three datasets created the possibility of building a model based on the genes of any chromosome to predict any clinical outcome. The preliminary models (discussed below) suggested little promise of interesting results to be derived from considering gene expression data in isolation, so we ultimately decided to focus on a more general predictive model involving more feature types instead.


**Sources**
[1] https://www.nia.nih.gov/health/alzheimers-disease-people-down-syndrome

[2] https://www.alz.org/alzheimers-dementia/what-is-dementia/types-of-dementia/down-syndrome

[3] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4019841/

https://www.ncbi.nlm.nih.gov/pubmed/18199027


http://www.bbc.com/future/story/20181022-there-is-mounting-evidence-that-herpes-leads-to-alzheimers





To determine the chromosomal location and biological role of any gene, we used this [gene annotation database provided by affymetrix](http://www.affymetrix.com/support/technical/byproduct.affx?product=HG-U219). Using this database, we can add columns to the gene expression profile dataset for the chromosomal column and gene name of every gene. From this, we can isolate genes based on their chromosomal location.



```python
annote = pd.read_csv("excel_annotations.csv", skiprows = 25)
```




```python
dropcolumns = ['GeneChip Array','Species Scientific Name','Annotation Date','Sequence Type','Sequence Source',
               'Transcript ID(Array Design)','Target Description','Representative Public ID',
               'Archival UniGene Cluster','Genome Version','Unigene Cluster Type','Ensembl','EC','OMIM',
              'FlyBase','AGI','WormBase','MGI Name','RGD Name','SGD accession number','QTL','Annotation Description',
              'Annotation Transcript Cluster','Transcript Assignments','Annotation Notes']
```




```python
annote = annote.drop(dropcolumns, axis = 1)
```




```python
Chromosome = []
for item in annote["Chromosomal Location"]:
    Chromosome.append(item.replace("chr","").replace('cen','').replace('-','').replace('p','q').split('q')[0])
annote['Chromosome']=Chromosome
```




```python
print("Chromosome 21 genes present in gene expression profile dataset: ",sum(annote['Chromosome']=='21'))

```


    Chromosome 21 genes present in gene expression profile dataset:  609




```python
GEP_df = pd.read_csv("ADNI_Gene_Expression_Profile.csv", low_memory=False, index_col = 'Phase')

ID_column = GEP_df.iloc[1].copy().values
ID_column[0] = 'LocusLink'
ID_column[1] = 'Symbol'
GEP_df.iloc[1] = GEP_df.columns
GEP_df.columns = ID_column
GEP_df.rename(index={'SubjectID':'Phase'}, inplace=True)

GEP_df = GEP_df.rename_axis("Gene_PSID")
GEP_df = GEP_df.rename_axis("SubjectID", axis = 'columns')

GEP_df.loc['Phase'][0] = np.nan
GEP_df.loc['Phase'][1] = np.nan

GEP_df.drop(['ProbeSet'])

nan = GEP_df.columns[-1]
GEP_df = GEP_df.rename(columns={nan:'Biological Name'})

```




```python
chrlist = [0]*len(GEP_df.index)
for i, gene in enumerate(GEP_df.index):
    gene_id = gene
    try:
        gene_ch = annote[annote['Probe Set ID']==gene_id]['Chromosome'].values[0]
        chrlist[i] = gene_ch
    except IndexError:
        chrlist[i] = np.nan
```




```python
GEP_df['Chromosome'] = chrlist
```


### Chromosome 21 Dataset



```python
GEP_head = GEP_df.iloc[0:8]
GEP_data = GEP_df.iloc[8:]
c21 = GEP_data[GEP_data['Chromosome']=='21']
GEP_C21 = pd.concat((GEP_head,c21))

display(GEP_C21.head(15))
```



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
      <th>SubjectID</th>
      <th>LocusLink</th>
      <th>Symbol</th>
      <th>116_S_1249</th>
      <th>037_S_4410</th>
      <th>006_S_4153</th>
      <th>116_S_1232</th>
      <th>099_S_4205</th>
      <th>007_S_4467</th>
      <th>128_S_0205</th>
      <th>003_S_2374</th>
      <th>...</th>
      <th>014_S_4668</th>
      <th>130_S_0289</th>
      <th>141_S_4456</th>
      <th>009_S_2381</th>
      <th>053_S_4557</th>
      <th>073_S_4300</th>
      <th>041_S_4014</th>
      <th>007_S_0101</th>
      <th>Biological Name</th>
      <th>Chromosome</th>
    </tr>
    <tr>
      <th>Gene_PSID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Visit</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>m48</td>
      <td>v03</td>
      <td>v03</td>
      <td>m48</td>
      <td>v03</td>
      <td>v03</td>
      <td>v06</td>
      <td>bl</td>
      <td>...</td>
      <td>v03</td>
      <td>m60</td>
      <td>v03</td>
      <td>bl</td>
      <td>v03</td>
      <td>v03</td>
      <td>v03</td>
      <td>v06</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Phase</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>ADNIGO</td>
      <td>ADNI2</td>
      <td>ADNI2.1</td>
      <td>ADNIGO.1</td>
      <td>ADNI2.2</td>
      <td>ADNI2.3</td>
      <td>ADNI2.4</td>
      <td>ADNIGO.2</td>
      <td>...</td>
      <td>ADNI2.443</td>
      <td>ADNIGO.293</td>
      <td>ADNI2.444</td>
      <td>ADNIGO.294</td>
      <td>ADNI2.445</td>
      <td>ADNI2.446</td>
      <td>ADNI2.447</td>
      <td>ADNI2.448</td>
      <td>Unnamed: 747</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>260/280</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.05</td>
      <td>2.07</td>
      <td>2.04</td>
      <td>2.03</td>
      <td>2.01</td>
      <td>2.05</td>
      <td>1.95</td>
      <td>1.99</td>
      <td>...</td>
      <td>2.05</td>
      <td>1.98</td>
      <td>2.09</td>
      <td>1.87</td>
      <td>2.03</td>
      <td>2.11</td>
      <td>1.94</td>
      <td>2.06</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>260/230</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.55</td>
      <td>1.54</td>
      <td>2.1</td>
      <td>1.52</td>
      <td>1.6</td>
      <td>1.91</td>
      <td>1.47</td>
      <td>2.07</td>
      <td>...</td>
      <td>2.05</td>
      <td>1.65</td>
      <td>1.56</td>
      <td>1.45</td>
      <td>1.33</td>
      <td>0.27</td>
      <td>1.72</td>
      <td>1.35</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>RIN</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.7</td>
      <td>7.6</td>
      <td>7.2</td>
      <td>6.8</td>
      <td>7.9</td>
      <td>7</td>
      <td>7.9</td>
      <td>7.2</td>
      <td>...</td>
      <td>6.5</td>
      <td>6.3</td>
      <td>6.4</td>
      <td>6.6</td>
      <td>6.8</td>
      <td>6.2</td>
      <td>5.8</td>
      <td>6.7</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Affy Plate</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>7</td>
      <td>3</td>
      <td>6</td>
      <td>7</td>
      <td>9</td>
      <td>4</td>
      <td>3</td>
      <td>8</td>
      <td>...</td>
      <td>6</td>
      <td>9</td>
      <td>3</td>
      <td>8</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>YearofCollection</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2011</td>
      <td>2012</td>
      <td>2011</td>
      <td>2011</td>
      <td>2011</td>
      <td>2012</td>
      <td>2011</td>
      <td>2011</td>
      <td>...</td>
      <td>2012</td>
      <td>2011</td>
      <td>2012</td>
      <td>2011</td>
      <td>2012</td>
      <td>2011</td>
      <td>2011</td>
      <td>2012</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ProbeSet</th>
      <td>LocusLink</td>
      <td>Symbol</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11715130_s_at</th>
      <td>LOC337967</td>
      <td>KRTAP6-2</td>
      <td>2.424</td>
      <td>2.623</td>
      <td>2.501</td>
      <td>3.103</td>
      <td>2.567</td>
      <td>2.992</td>
      <td>2.249</td>
      <td>2.63</td>
      <td>...</td>
      <td>3.074</td>
      <td>2.763</td>
      <td>2.89</td>
      <td>3.155</td>
      <td>2.526</td>
      <td>2.452</td>
      <td>2.822</td>
      <td>2.651</td>
      <td>[KRTAP6-2] keratin associated protein 6-2</td>
      <td>21</td>
    </tr>
    <tr>
      <th>11715131_s_at</th>
      <td>LOC337975</td>
      <td>KRTAP20-1</td>
      <td>1.826</td>
      <td>2.306</td>
      <td>2.735</td>
      <td>2.777</td>
      <td>2.897</td>
      <td>2.631</td>
      <td>2.613</td>
      <td>2.481</td>
      <td>...</td>
      <td>2.46</td>
      <td>2.374</td>
      <td>2.074</td>
      <td>3.121</td>
      <td>2.725</td>
      <td>2.21</td>
      <td>2.828</td>
      <td>2.444</td>
      <td>[KRTAP20-1] keratin associated protein 20-1</td>
      <td>21</td>
    </tr>
    <tr>
      <th>11715144_s_at</th>
      <td>LOC337974</td>
      <td>KRTAP19-7</td>
      <td>3.256</td>
      <td>3.404</td>
      <td>4.112</td>
      <td>3.279</td>
      <td>3.67</td>
      <td>3.279</td>
      <td>3.257</td>
      <td>3.86</td>
      <td>...</td>
      <td>3.856</td>
      <td>3.64</td>
      <td>3.642</td>
      <td>3.772</td>
      <td>4.187</td>
      <td>3.579</td>
      <td>4.542</td>
      <td>3.92</td>
      <td>[KRTAP19-7] keratin associated protein 19-7</td>
      <td>21</td>
    </tr>
    <tr>
      <th>11715145_s_at</th>
      <td>LOC337976</td>
      <td>KRTAP20-2</td>
      <td>3.529</td>
      <td>4.029</td>
      <td>4.254</td>
      <td>4.094</td>
      <td>3.629</td>
      <td>3.632</td>
      <td>3.614</td>
      <td>4.021</td>
      <td>...</td>
      <td>4.026</td>
      <td>3.808</td>
      <td>3.966</td>
      <td>4.134</td>
      <td>3.881</td>
      <td>3.833</td>
      <td>4.112</td>
      <td>3.945</td>
      <td>[KRTAP20-2] keratin associated protein 20-2</td>
      <td>21</td>
    </tr>
    <tr>
      <th>11715156_s_at</th>
      <td>LOC337966</td>
      <td>KRTAP6-1</td>
      <td>3.855</td>
      <td>3.9</td>
      <td>4.124</td>
      <td>4.428</td>
      <td>3.947</td>
      <td>4.371</td>
      <td>4.14</td>
      <td>4.129</td>
      <td>...</td>
      <td>4.783</td>
      <td>3.961</td>
      <td>4.035</td>
      <td>4.428</td>
      <td>3.972</td>
      <td>4.208</td>
      <td>4.622</td>
      <td>4.147</td>
      <td>[KRTAP6-1] keratin associated protein 6-1</td>
      <td>21</td>
    </tr>
    <tr>
      <th>11715157_s_at</th>
      <td>LOC337969</td>
      <td>KRTAP19-2</td>
      <td>2</td>
      <td>2.162</td>
      <td>2.135</td>
      <td>2.144</td>
      <td>2.144</td>
      <td>2.147</td>
      <td>1.938</td>
      <td>2.27</td>
      <td>...</td>
      <td>2.19</td>
      <td>2.045</td>
      <td>2.545</td>
      <td>2.222</td>
      <td>2.332</td>
      <td>1.998</td>
      <td>2.133</td>
      <td>2.238</td>
      <td>[KRTAP19-2] keratin associated protein 19-2</td>
      <td>21</td>
    </tr>
    <tr>
      <th>11715158_s_at</th>
      <td>LOC337971</td>
      <td>KRTAP19-4</td>
      <td>2.682</td>
      <td>2.993</td>
      <td>2.778</td>
      <td>2.904</td>
      <td>2.714</td>
      <td>2.672</td>
      <td>2.837</td>
      <td>2.578</td>
      <td>...</td>
      <td>2.617</td>
      <td>2.877</td>
      <td>2.944</td>
      <td>2.612</td>
      <td>2.729</td>
      <td>2.7</td>
      <td>2.837</td>
      <td>2.582</td>
      <td>[KRTAP19-4] Keratin associated protein 19-4</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
<p>15 rows × 748 columns</p>
</div>




```python

```




```python
adni_df = pd.read_csv("ADNIMERGE.csv", low_memory=False)
```




```python
keep_columns = ['PTID','DX','Month']
outcome_df = adni_df[keep_columns]
```




```python
"""
if any diagnoses are dementia, report dementia - if all diagnoses ara NaN, report NaN.
"""


n = len(GEP_C21.columns[2:-2])
target = np.full(n,0)
for i, col in enumerate(GEP_C21.columns[2:-2]):
    if (outcome_df[outcome_df['PTID']==col]['DX'] == 'Dementia').any():
        target[i] = 1
    elif (pd.Series([str(o) == 'nan' for o in outcome_df[outcome_df['PTID']==col]['DX']])).all():
        target[i] = np.nan

```


From this dataset, we find 13 features corresponding to probes of the APP gene, a major implicated gene in Alzheimer's Disease.

To analyse the role of different genes on chromosome 21 in the etiology of Alzheimer's Disease, a good baseline model would be to use these 13 features in a simple baseline model, so we fitted a simple logistic regression to an X data set of these 13 features.

To ensure that imbalance in the data was not an issue, we used Synthetic Minority Over-sampling Technique (SMOTE) using the imblearn package.



```python
APP_PSID = annote[annote['Gene Symbol'] == 'APP']['Probe Set ID'].values

no_markers = len(APP_PSID)
no_patients = len(GEP_C21.columns[2:-2])
```




```python
X = np.zeros((no_patients,no_markers))

for i, col in enumerate(GEP_C21.columns[2:-2]):
    patient = GEP_C21[col]
    for j, PSID in enumerate(APP_PSID):
        X[i,j] = patient[PSID]
```




```python
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3, random_state=42)
```




```python
sm =SMOTE(ratio=1, random_state=42)
X_TR, y_TR = sm.fit_resample(X_train,y_train)
```




```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
```




```python
logreg = LogisticRegressionCV(cv=5, max_iter=10000).fit(X_TR,y_TR)
```




```python
app_crossvalscore = np.mean(cross_val_score(logreg,X_TR,y_TR, cv=5))
```




```python
app_testscore = accuracy_score(logreg.predict(X_test),y_test)
```


The cross-validation and test scores for this baseline model are:



```python
print("Cross-validation score ",app_crossvalscore)
print("Test score ",app_testscore)
```


    Cross-validation score  0.5798706240487063
    Test score  0.5178571428571429


Both appear only marginally better than random. APP gene expression profile appears to be a very poor predictor of AD.

Next, we looked to build a similar simplistic model using all genes on chromosome 21 to see whether there was anything in particular that stood out as interesting.



```python
PSIDs = GEP_C21.index[8:].values

no_markers2 = len(PSIDs)
no_patients2 = len(GEP_C21.columns[2:-2])

```




```python
X2 = np.zeros((no_patients2,no_markers2))

for i, col in enumerate(GEP_C21.columns[2:-2]):
    patient = GEP_C21[col]
    for j, PSID in enumerate(PSIDs):
        X2[i,j] = patient[PSID]


```




```python
X_tra, X_tst, y_tra, y_tst = train_test_split(X2,target, test_size=0.3, random_state=42)
```




```python
sm = SMOTE(ratio=1, random_state=42)
X_TR2, y_TR2 = sm.fit_resample(X_tra,y_tra)
```




```python
logreg2 = LogisticRegressionCV(cv=5, max_iter=10000).fit(X_TR2,y_TR2)
```




```python
c21_crossvalscore = np.mean(cross_val_score(logreg2,X_TR2,y_TR2, cv=5))
```




```python
c21_testscore = np.mean(cross_val_score(logreg2,X_tst,y_tst, cv=5))
```


The results of this are:



```python
print("Cross-validation score ",c21_crossvalscore)
print("Test score ",c21_testscore)
```


    Cross-validation score  0.8115677321156773
    Test score  0.6965041721563461


These results are slightly more promising, and warrant an investigation into the details of the model: what are the genes with the largest magnitude associated coefficient?



```python
big = np.argmax(logreg2.coef_)
small = np.argmin(logreg2.coef_)
big_gene = PSIDs[big]
small_gene = PSIDs[small]
```




```python
print("Biggest positive coefficient: ")
annote[annote['Probe Set ID']==big_gene]
```


    Biggest positive coefficient:





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
      <th>Probe Set ID</th>
      <th>UniGene ID</th>
      <th>Alignments</th>
      <th>Gene Title</th>
      <th>Gene Symbol</th>
      <th>Chromosomal Location</th>
      <th>Entrez Gene</th>
      <th>SwissProt</th>
      <th>RefSeq Protein ID</th>
      <th>RefSeq Transcript ID</th>
      <th>Gene Ontology Biological Process</th>
      <th>Gene Ontology Cellular Component</th>
      <th>Gene Ontology Molecular Function</th>
      <th>Pathway</th>
      <th>InterPro</th>
      <th>Trans Membrane</th>
      <th>Chromosome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>133</th>
      <td>11715233_s_at</td>
      <td>Hs.381214</td>
      <td>chr21:47581062-47604373 (-) // 95.94 // q22.3</td>
      <td>spermatogenesis and centriole associated 1-like</td>
      <td>SPATC1L</td>
      <td>chr21q22.3</td>
      <td>84221</td>
      <td>Q9H0A9</td>
      <td>NP_001136326 /// NP_115637 /// XP_005261245 //...</td>
      <td>NM_001142854 /// NM_032261 /// XM_005261188 //...</td>
      <td>---</td>
      <td>---</td>
      <td>0005515 // protein binding // inferred from ph...</td>
      <td>---</td>
      <td>IPR029384 // Speriolin, C-terminal // 1.0E-75 ...</td>
      <td>---</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>





```python
print("Biggest negative coefficient: ")
annote[annote['Probe Set ID']==small_gene]
```


    Biggest negative coefficient:





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
      <th>Probe Set ID</th>
      <th>UniGene ID</th>
      <th>Alignments</th>
      <th>Gene Title</th>
      <th>Gene Symbol</th>
      <th>Chromosomal Location</th>
      <th>Entrez Gene</th>
      <th>SwissProt</th>
      <th>RefSeq Protein ID</th>
      <th>RefSeq Transcript ID</th>
      <th>Gene Ontology Biological Process</th>
      <th>Gene Ontology Cellular Component</th>
      <th>Gene Ontology Molecular Function</th>
      <th>Pathway</th>
      <th>InterPro</th>
      <th>Trans Membrane</th>
      <th>Chromosome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8325</th>
      <td>11723425_at</td>
      <td>Hs.529400</td>
      <td>chr21:34697208-34732236 (+) // 82.3 // q22.11</td>
      <td>interferon (alpha, beta and omega) receptor 1</td>
      <td>IFNAR1</td>
      <td>chr21q22.11</td>
      <td>3454</td>
      <td>P17181</td>
      <td>NP_000620 /// XP_005261021 /// XP_011527854</td>
      <td>NM_000629 /// XM_005260964 /// XM_011529552</td>
      <td>0007166 // cell surface receptor signaling pat...</td>
      <td>0005622 // intracellular // traceable author s...</td>
      <td>0004904 // interferon receptor activity // inf...</td>
      <td>---</td>
      <td>IPR003961 // Fibronectin type III // 2.1E-35 /...</td>
      <td>---</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>



Here, the gene with the largest positive coefficient in this model has no identified link with cognitive function, autophagy, protein synthesis, calcium signalling, immune system, inflammation or any other process linked with neurodegenerative disease - and is only moderately expressed in the brain. Moreover, the gene with the strongest negative coefficient has actually been previously been positively linked with amyloidogenesis in mice, contrary to what our model might imply.

This suggests that there may be limited immediate benefit to using gene expression profile data. To further consider this, we plotted the gene expression distribution for all 13 APP gene probes and also the gene probe found to be the strongest predictor for our model - we see no clear difference in distribution between AD and non-AD, implying again that gene expression data may not be as useful as first thought.



```python
xpos = X_TR[y_TR==1]
xneg = X_TR[y_TR==0]
plt.subplots(3,4, figsize=(16,8))
plt.suptitle("Amyloid Precursor Protein Gene Expression Profile for Different Microarray Probes", y=1.04, fontsize =22)
for i in range(12):
    ax = plt.subplot(3,4,i+1)
    ax.set_title("Microarray Probe {0}".format(str(i+1)))
    sns.distplot(xpos[i], bins=30, color = 'red', ax = ax, hist=False, label = "AD")
    sns.distplot(xneg[i], bins=30, color = 'blue', ax = ax, hist=False, label = 'Not AD')
    ax.set_xlabel("Probe Expression Level")
    ax.set_ylabel("Density")
    ax.set_ylim((0,1))
plt.tight_layout()
plt.show()
```


    /anaconda3/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



![png](gene_expression_files/gene_expression_44_1.png)




```python
bigboy = X_TR2[:,13]

bigyes = bigboy[y_TR2==1]
bignah = bigboy[y_TR2==0]
plt.figure(figsize=(16,8))
sns.distplot(bigyes, bins=30, color = 'red', hist=False, label = "AD")
sns.distplot(bignah, bins=30, color = 'blue', hist=False, label = 'Not AD')
plt.title("Gene Expression Profile: Best Predictor in Logistic Model", fontsize=22)
plt.ylim((0,2))
plt.xlabel("Gene Expression Level")
plt.ylabel("Density")
plt.show()
```


    /anaconda3/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



![png](gene_expression_files/gene_expression_45_1.png)


When researching details of the gene expression dataset, it was found the the samples from which the data is derived were taken from blood, not the central nervous system. This means that the gene expression profiles are not indicative of the cellular environment in the brain, as the gene expression profiles of different tissues varies hugely and the central nervous system is separated from the rest of the body by a near-impervious blood-brain-barrier. As such, even though other models could be tested and different chromosomes could be examined, it was decided that the chance of finding valuable data was too low, and as such the group focussed on other aims.
