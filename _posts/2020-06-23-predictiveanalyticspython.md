---
title: "Using ML to Determine Infant Heart Rate Tracings - Python Portion (Part 2)"
date: 2020-06-23
classes: wide
header:
  image: "/images/predictiveanalytics/Number1.jpg"
excerpt: "Predictive Analytics, KNN, Decision Tree, Random Forest"
---
__Background__

Fetal heart rate monitoring as an indicator of fetal well-being can be inaccurate as predictors of a poor neonatal outcome and come with significant healthcare and medicolegal costs.  It is the goal of this project to use a database of specific technical characteristics of fetal heart rate monitoring from the UCI machine learning database to develop a predictive model using an automated system to better identify worrisome decreases in fetal heart rate.

The data that I will primarily use is from the University of California – Irvine Machine Learning Repository.  The dataset can be found at the following web address: [https://archive.ics.uci.edu/ml/datasets/Cardiotocography](https://archive.ics.uci.edu/ml/datasets/Cardiotocography).

According to the website, there were over 2000 fetal heart tracings (cardiotocograms) and interpreted by three expert obstetricians.  Many of the measurements include the technical measurements include heart rate accelerations, decelerations, max heart rate, minimum heart rates, heart rate baseline and finally the target variable is whether the tracing was normal, suspect, or pathologic.


```python
#Loading Our Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading Dataset
df = pd.read_excel("Fetal Monitoring.xls", sheet_name = "Raw Data").dropna()

#Dropping Identifiers
df = df.drop(columns = "FileName")
df = df.drop(columns = "Date")
df = df.drop(columns = "SegFile")

#Viewing Dataframe
df
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
      <th></th>
      <th>b</th>
      <th>e</th>
      <th>LBE</th>
      <th>LB</th>
      <th>AC</th>
      <th>FM</th>
      <th>UC</th>
      <th>ASTV</th>
      <th>MSTV</th>
      <th>ALTV</th>
      <th>MLTV</th>
      <th>DL</th>
      <th>DS</th>
      <th>DP</th>
      <th>DR</th>
      <th>Width</th>
      <th>Min</th>
      <th>Max</th>
      <th>Nmax</th>
      <th>Nzeros</th>
      <th>Mode</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Variance</th>
      <th>Tendency</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>AD</th>
      <th>DE</th>
      <th>LD</th>
      <th>FS</th>
      <th>SUSP</th>
      <th>CLASS</th>
      <th>NSP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>240.0</td>
      <td>357.0</td>
      <td>120.0</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>73.0</td>
      <td>0.5</td>
      <td>43.0</td>
      <td>2.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>64.0</td>
      <td>62.0</td>
      <td>126.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>137.0</td>
      <td>121.0</td>
      <td>73.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0</td>
      <td>632.0</td>
      <td>132.0</td>
      <td>132.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>17.0</td>
      <td>2.1</td>
      <td>0.0</td>
      <td>10.4</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>130.0</td>
      <td>68.0</td>
      <td>198.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>141.0</td>
      <td>136.0</td>
      <td>140.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>177.0</td>
      <td>779.0</td>
      <td>133.0</td>
      <td>133.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>16.0</td>
      <td>2.1</td>
      <td>0.0</td>
      <td>13.4</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>130.0</td>
      <td>68.0</td>
      <td>198.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>141.0</td>
      <td>135.0</td>
      <td>138.0</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>411.0</td>
      <td>1192.0</td>
      <td>134.0</td>
      <td>134.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>16.0</td>
      <td>2.4</td>
      <td>0.0</td>
      <td>23.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>53.0</td>
      <td>170.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>137.0</td>
      <td>134.0</td>
      <td>137.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>533.0</td>
      <td>1147.0</td>
      <td>132.0</td>
      <td>132.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>16.0</td>
      <td>2.4</td>
      <td>0.0</td>
      <td>19.9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>53.0</td>
      <td>170.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>137.0</td>
      <td>136.0</td>
      <td>138.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2122</th>
      <td>2059.0</td>
      <td>2867.0</td>
      <td>140.0</td>
      <td>140.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>79.0</td>
      <td>0.2</td>
      <td>25.0</td>
      <td>7.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>137.0</td>
      <td>177.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>153.0</td>
      <td>150.0</td>
      <td>152.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2123</th>
      <td>1576.0</td>
      <td>2867.0</td>
      <td>140.0</td>
      <td>140.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>78.0</td>
      <td>0.4</td>
      <td>22.0</td>
      <td>7.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>66.0</td>
      <td>103.0</td>
      <td>169.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>152.0</td>
      <td>148.0</td>
      <td>151.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2124</th>
      <td>1576.0</td>
      <td>2596.0</td>
      <td>140.0</td>
      <td>140.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>79.0</td>
      <td>0.4</td>
      <td>20.0</td>
      <td>6.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>67.0</td>
      <td>103.0</td>
      <td>170.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>153.0</td>
      <td>148.0</td>
      <td>152.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2125</th>
      <td>1576.0</td>
      <td>3049.0</td>
      <td>140.0</td>
      <td>140.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>78.0</td>
      <td>0.4</td>
      <td>27.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>66.0</td>
      <td>103.0</td>
      <td>169.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>152.0</td>
      <td>147.0</td>
      <td>151.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2126</th>
      <td>2796.0</td>
      <td>3415.0</td>
      <td>142.0</td>
      <td>142.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>74.0</td>
      <td>0.4</td>
      <td>36.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>42.0</td>
      <td>117.0</td>
      <td>159.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>145.0</td>
      <td>143.0</td>
      <td>145.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>2126 rows × 37 columns</p>
</div>




```python
#Checking Datatypes
df.dtypes
```




    b           float64
    e           float64
    LBE         float64
    LB          float64
    AC          float64
    FM          float64
    UC          float64
    ASTV        float64
    MSTV        float64
    ALTV        float64
    MLTV        float64
    DL          float64
    DS          float64
    DP          float64
    DR          float64
    Width       float64
    Min         float64
    Max         float64
    Nmax        float64
    Nzeros      float64
    Mode        float64
    Mean        float64
    Median      float64
    Variance    float64
    Tendency    float64
    A           float64
    B           float64
    C           float64
    D           float64
    E           float64
    AD          float64
    DE          float64
    LD          float64
    FS          float64
    SUSP        float64
    CLASS       float64
    NSP         float64
    dtype: object




```python
#Re-coding Categorical Variables
df['NSP'] = df['NSP'].astype('category')
df['Tendency'] = df['Tendency'].astype('category')
df['A'] = df['A'].astype('category')
df['B'] = df['B'].astype('category')
df['C'] = df['C'].astype('category')
df['D'] = df['D'].astype('category')
df['E'] = df['E'].astype('category')
df['AD'] = df['AD'].astype('category')
df['DE'] = df['DE'].astype('category')
df['LD'] = df['LD'].astype('category')
df['SUSP'] = df['SUSP'].astype('category')
df['CLASS'] = df['CLASS'].astype('category')

#Confirming were encoded correctly
df.dtypes
```




    b            float64
    e            float64
    LBE          float64
    LB           float64
    AC           float64
    FM           float64
    UC           float64
    ASTV         float64
    MSTV         float64
    ALTV         float64
    MLTV         float64
    DL           float64
    DS           float64
    DP           float64
    DR           float64
    Width        float64
    Min          float64
    Max          float64
    Nmax         float64
    Nzeros       float64
    Mode         float64
    Mean         float64
    Median       float64
    Variance     float64
    Tendency    category
    A           category
    B           category
    C           category
    D           category
    E           category
    AD          category
    DE          category
    LD          category
    FS           float64
    SUSP        category
    CLASS       category
    NSP         category
    dtype: object



__Description of Variables__

FileName and SegFile = Personal Identifiers

Date = Individual Date of Measurement

b = Starting Point of Measurement

e = Ending Point of Measurement

LBE = Baseline value (By Medical Expert)

LB = Fetal Heart Rate Baseline (beats/minute - Automated)

AC = # Accelerations/second

FM = # Fetal Movements/second

UC = # Uterine Contractions/second

DL = # Light Decelerations/second

DS = # Severe Decelerations/second

DP = # Prolonged Decelerations/second

DR = # Repetitive Decelerations (All 0s)

ASTV = % of time with abnormal short term variability

MSTV = Mean Value of Short Term Variability

ALTV = % of time with abnormal long term variability

MLTV = Mean Value of Long Term Variability

Width = Width of Fetal Heart Rate Histogram

Min = Minimum of Fetal Heart Rate Histogram

Max = Maximum of Fetal Heart Rate Histogram

Nmax = # Histogram Peaks

Nzeros = # Histogram Zeros

Mode = Histogram Mode

Mean = Histogram Mean

Median = Histogram Median

Variance = Histogram Variance

Tendency = Histogram Tendency:  -1 = Left Asymmetric, 0 = Symmetric, 1 = Right Asymmetric

A = Calm Sleep

B = REM Sleep

C = Calm Vigilance

D = Active Vigilance

AD = Accelerative/Decelerative Pattern (Stress Situation)

DE = Decelerative Pattern (Vagal Stimulation)

LD = Largely decelerative pattern

FS = Flat Sinusoidal Pattern (Pathologic State)

SUSP = Suspect Pattern

Class = 1 to 10 for Classes A to SUSP

NSP = Fetal State Class Code (Normal = 1, Suspect = 2, Pathologic = 3)

__In the R portion of this project, high levels of correlation were found between the b and e variables, the expert and automated determination of fetal heart rate, and finally mode, mean, and median variables were all highly correlated with one another.__

__In this dataset, there is a 10 label multi-class target as well as the 3 label multi-class target variable.  I will be using the 3 label multi-class target variable for my target.__  

__The target variables is NSP which classifies the FHR as Normal, Suspect, or Pathologic.__

__However, I will first include the 10 label target variables as predictors to see if this increases model performance after hyperparameter tuning and then I will do this without the 10 class variables to see if the performance is better or worse.__

__I plan on checking K Nearest Neighbors, Decision Tree, and Random Forest algorithms to determine the best model.__


```python
#Checking Features for High Correlations With Correlation >
corr_matrix = df.corr().abs()

#Selecting Upper Triangle of Correlation Matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                 k=1).astype(np.bool))

#Find index of feature columns with correlation >0.90
to_drop = [column for column in upper.columns if any(upper[column] >0.90)]
to_drop
```




    ['e', 'LB', 'Median']




```python
#Dropping Columns with High Correlations >0.90
#I will actually drop LBE since this is the expert determined variable.  We want to test the applicability of the SIS Porto system so we will
#keep the LB (automated) one.

#Dropping 'LB' from our Dataframe
df = df.drop(columns = 'LBE')
df = df.drop(columns = 'e')
df = df.drop(columns = 'Median')
```


```python
pd.set_option('display.max_columns', 500)
df
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
      <th></th>
      <th>b</th>
      <th>LB</th>
      <th>AC</th>
      <th>FM</th>
      <th>UC</th>
      <th>ASTV</th>
      <th>MSTV</th>
      <th>ALTV</th>
      <th>MLTV</th>
      <th>DL</th>
      <th>DS</th>
      <th>DP</th>
      <th>DR</th>
      <th>Width</th>
      <th>Min</th>
      <th>Max</th>
      <th>Nmax</th>
      <th>Nzeros</th>
      <th>Mode</th>
      <th>Mean</th>
      <th>Variance</th>
      <th>Tendency</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>AD</th>
      <th>DE</th>
      <th>LD</th>
      <th>FS</th>
      <th>SUSP</th>
      <th>CLASS</th>
      <th>NSP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>240.0</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>73.0</td>
      <td>0.5</td>
      <td>43.0</td>
      <td>2.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>64.0</td>
      <td>62.0</td>
      <td>126.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>137.0</td>
      <td>73.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0</td>
      <td>132.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>17.0</td>
      <td>2.1</td>
      <td>0.0</td>
      <td>10.4</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>130.0</td>
      <td>68.0</td>
      <td>198.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>141.0</td>
      <td>136.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>177.0</td>
      <td>133.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>16.0</td>
      <td>2.1</td>
      <td>0.0</td>
      <td>13.4</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>130.0</td>
      <td>68.0</td>
      <td>198.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>141.0</td>
      <td>135.0</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>411.0</td>
      <td>134.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>16.0</td>
      <td>2.4</td>
      <td>0.0</td>
      <td>23.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>53.0</td>
      <td>170.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>137.0</td>
      <td>134.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>533.0</td>
      <td>132.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>16.0</td>
      <td>2.4</td>
      <td>0.0</td>
      <td>19.9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>53.0</td>
      <td>170.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>137.0</td>
      <td>136.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2122</th>
      <td>2059.0</td>
      <td>140.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>79.0</td>
      <td>0.2</td>
      <td>25.0</td>
      <td>7.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>137.0</td>
      <td>177.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>153.0</td>
      <td>150.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2123</th>
      <td>1576.0</td>
      <td>140.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>78.0</td>
      <td>0.4</td>
      <td>22.0</td>
      <td>7.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>66.0</td>
      <td>103.0</td>
      <td>169.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>152.0</td>
      <td>148.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2124</th>
      <td>1576.0</td>
      <td>140.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>79.0</td>
      <td>0.4</td>
      <td>20.0</td>
      <td>6.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>67.0</td>
      <td>103.0</td>
      <td>170.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>153.0</td>
      <td>148.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2125</th>
      <td>1576.0</td>
      <td>140.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>78.0</td>
      <td>0.4</td>
      <td>27.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>66.0</td>
      <td>103.0</td>
      <td>169.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>152.0</td>
      <td>147.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2126</th>
      <td>2796.0</td>
      <td>142.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>74.0</td>
      <td>0.4</td>
      <td>36.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>42.0</td>
      <td>117.0</td>
      <td>159.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>145.0</td>
      <td>143.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>2126 rows × 34 columns</p>
</div>




```python
#Setting Up Features and Target Variables
target = df['NSP']
features = df.loc[:, df.columns != 'NSP']

#Getting Dummy Variables for our categorical variables
target = pd.get_dummies(target)
features = pd.get_dummies(features)

#Importing Packages for KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score

#Create Standardizer
standardizer = StandardScaler()

#Standardize Features
features_standardized = standardizer.fit_transform(features)

#Train/Test 80/20 Split
features_train, features_test, target_train, target_test = train_test_split(features_standardized, target, test_size = 0.2, random_state = 1)

#Creating Classifier with K of 3
knn = KNeighborsClassifier(n_neighbors = 3, n_jobs = -1)

#Fitting Classifier on Trianing Data
knn.fit(features_train, target_train)
print(knn.fit)

#Generating Confusion Matrix
target_pred = knn.predict(features_test)
test0 = np.array(target_test).argmax(axis = 1)
predictions0 = np.array(target_pred).argmax(axis = 1)
print(confusion_matrix(test0, predictions0))

#Printing Classification Report for KNN with N-Neighbors of 3
print(classification_report(test0, predictions0))
```

    <bound method SupervisedIntegerMixin.fit of KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=-1, n_neighbors=3, p=2,
                         weights='uniform')>
    [[326   0   0]
     [  1  67   0]
     [  0   1  31]]
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00       326
               1       0.99      0.99      0.99        68
               2       1.00      0.97      0.98        32

        accuracy                           1.00       426
       macro avg       0.99      0.98      0.99       426
    weighted avg       1.00      1.00      1.00       426




```python
#Hyperparameter Tuning for KNN Using Grid Search CV

#Creating Hyperparameter Grid
param_dist1 = {"leaf_size": list(range(1,50)),
              "n_neighbors": list(range(1,30)),
              "p": [1,2]}

#Create New KNN Object
knn_2 = KNeighborsClassifier()

#Use Gridsearch
clf = GridSearchCV(knn_2, param_dist1, cv=10, n_jobs=-1)

#Fit Model
best_model = clf.fit(features_standardized, target)

print('Best Leaf Size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
print('Best Metric:', best_model.best_estimator_.get_params()['metric'])
print('Best Weights:', best_model.best_estimator_.get_params()['weights'])
```

    Best Leaf Size: 1
    Best p: 1
    Best n_neighbors: 3
    Best Metric: minkowski
    Best Weights: uniform



```python
#Checking Best KNN Model Using Leaf Size of 1, P of 1, and Best Neighbors of 3, Uniform Weights

#Setting Up Features and Target Variables
target = df['NSP']
features = df.loc[:, df.columns != 'NSP']

#Getting Dummy Variables for our categorical variables
target = pd.get_dummies(target)
features = pd.get_dummies(features)

#Create Standardizer
standardizer = StandardScaler()

#Standardize Features
features_standardized = standardizer.fit_transform(features)

#Train/Test 80/20 Split
features_train, features_test, target_train, target_test = train_test_split(features_standardized, target, test_size = 0.2, random_state = 1)

#Creating Classifier with K of 3
knn = KNeighborsClassifier(n_neighbors = 3, n_jobs = -1, p = 1, leaf_size = 1, weights = "uniform", metric = "minkowski")

#Fitting Classifier on Trianing Data
knn.fit(features_train, target_train)

#Printing Classification Report
target_pred = knn.predict(features_test)
test0 = np.array(target_test).argmax(axis = 1)
predictions0 = np.array(target_pred).argmax(axis = 1)
print(confusion_matrix(test0, predictions0))

#Printing Classification Report
print(classification_report(test0, predictions0))
```

    [[326   0   0]
     [  2  66   0]
     [  0   1  31]]
                  precision    recall  f1-score   support

               0       0.99      1.00      1.00       326
               1       0.99      0.97      0.98        68
               2       1.00      0.97      0.98        32

        accuracy                           0.99       426
       macro avg       0.99      0.98      0.99       426
    weighted avg       0.99      0.99      0.99       426




```python
#Fitting Data On Decision Tree Algorithm

#Importing Packages
from sklearn.tree import DecisionTreeClassifier

#Setting Up Features and Target Variables
target = df['NSP']
features = df.loc[:, df.columns != 'NSP']

#Train/Test Splitting with 80/20 Split
features_train1, features_test1, target_train1, target_test1 = train_test_split(features, target, test_size = 0.2, random_state = 1)

#Instantiating Decision Tree Classifier
dt = DecisionTreeClassifier(random_state = 1)
print(dt.fit(features_train1, target_train1))

#Setting Target Prediction Variable Based on Features Test
target_pred = dt.predict(features_test1)

#Generating Confusion Matrix
test1 = np.array(target_test1)
predictions1 = np.array(target_pred)
print(confusion_matrix(test1, predictions1))

#Classification Report for Decision Tree
print(classification_report(test1, predictions1))
```

    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=1, splitter='best')
    [[324   2   0]
     [  1  67   0]
     [  0   1  31]]
                  precision    recall  f1-score   support

             1.0       1.00      0.99      1.00       326
             2.0       0.96      0.99      0.97        68
             3.0       1.00      0.97      0.98        32

        accuracy                           0.99       426
       macro avg       0.98      0.98      0.98       426
    weighted avg       0.99      0.99      0.99       426




```python
#Hyperparameter Tuning for Decision Tree

#Creating Hyperparameter Grid
param_dist = {"max_depth": [3, None],
             "criterion": ["gini", "entropy"]}

tree_cv = RandomizedSearchCV(dt, param_dist, cv = 10)

tree_cv.fit(features_train1, target_train1)

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_estimator_))
print("Best Score is {}".format(tree_cv.best_score_))
```

    C:\Users\blmay\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:266: UserWarning: The total space of parameters 4 is smaller than n_iter=10. Running 4 iterations. For exhaustive searches, use GridSearchCV.
      % (grid_size, self.n_iter, grid_size), UserWarning)


    Tuned Decision Tree Parameters: DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=1, splitter='best')
    Best Score is 0.9847058823529412



```python
#Fitting New DT Algorithm on Tuned Hyperparameters

#Setting Up Features and Target Variables
target = df['NSP']
features = df.loc[:, df.columns != 'NSP']

#Train/Test Splitting with 80/20 Split
features_train1, features_test1, target_train1, target_test1 = train_test_split(features, target, test_size = 0.2, random_state = 1)

#Instantiating Decision Tree Classifier
dt = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=1, splitter='best')
dt.fit(features_train1, target_train1)

#Setting Target Prediction Variable Based on Features Test
target_pred = dt.predict(features_test1)

#Generating Confusion Matrix
test1 = np.array(target_test1)
predictions1 = np.array(target_pred)
print(confusion_matrix(test1, predictions1))

#Classification Report for Decision Tree
print(classification_report(test1, predictions1))
```

    [[321   5   0]
     [  2  66   0]
     [  0   1  31]]
                  precision    recall  f1-score   support

             1.0       0.99      0.98      0.99       326
             2.0       0.92      0.97      0.94        68
             3.0       1.00      0.97      0.98        32

        accuracy                           0.98       426
       macro avg       0.97      0.97      0.97       426
    weighted avg       0.98      0.98      0.98       426




```python
#Fitting Data on Random Forest Algorithm

#Importing Packages
from sklearn.ensemble import RandomForestClassifier

#Setting Up Features and Target Variables
target = df['NSP']
features = df.loc[:, df.columns != 'NSP']

#Train/Test Splitting with 80/20 Split
features_train2, features_test2, target_train2, target_test2 = train_test_split(features, target, test_size = 0.2, random_state = 1)

#Creating RFC Model
model = RandomForestClassifier(n_estimators = 100, random_state = 1)

#Fitting Training Data
print(model.fit(features_train2, target_train2))

#Checking Predictions
rf_predictions = model.predict(features_test2)

#Generating Confusion Matrix
test2 = np.array(target_test2)
predictions2 = np.array(rf_predictions)
print(confusion_matrix(test2, predictions2))

#Classification Report
print(classification_report(test2, predictions2))
```

    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=1, verbose=0,
                           warm_start=False)
    [[326   0   0]
     [  2  66   0]
     [  0   1  31]]
                  precision    recall  f1-score   support

             1.0       0.99      1.00      1.00       326
             2.0       0.99      0.97      0.98        68
             3.0       1.00      0.97      0.98        32

        accuracy                           0.99       426
       macro avg       0.99      0.98      0.99       426
    weighted avg       0.99      0.99      0.99       426




```python
#Hyperparameter Tuning for Random Forest Algorithm

#Number of tress in Random Forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]

#Number of Features to Consider At Each Split
max_features = ['auto', 'sqrt']

#Maximum Number of Levels in Tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

#Minimum Number of Samples Required At Each Leaf
min_samples_split = [2,5,10]

#Minimum Number of Samles Required at Each Leaf Node
min_samples_leaf = [1,2,4]

#Method of Selecting Samples for Training Each Tree
bootstrap = [True, False]

#Create Random Grid
random_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap}

#Using Random Grid to Search for Best Hyperparameters

rf_random1 = RandomForestClassifier()

rf_random2 = RandomizedSearchCV(estimator = rf_random1, param_distributions = random_grid, n_iter = 100, cv = 10, verbose = 2, random_state = 1, n_jobs = -1)

#Fitting the Model
rf_random2.fit(features, target)
```

    Fitting 10 folds for each of 100 candidates, totalling 1000 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  17 tasks      | elapsed:    5.3s
    [Parallel(n_jobs=-1)]: Done 138 tasks      | elapsed:   31.4s
    [Parallel(n_jobs=-1)]: Done 341 tasks      | elapsed:  1.5min
    [Parallel(n_jobs=-1)]: Done 624 tasks      | elapsed:  2.8min
    [Parallel(n_jobs=-1)]: Done 1000 out of 1000 | elapsed:  4.6min finished





    RandomizedSearchCV(cv=10, error_score='raise-deprecating',
                       estimator=RandomForestClassifier(bootstrap=True,
                                                        class_weight=None,
                                                        criterion='gini',
                                                        max_depth=None,
                                                        max_features='auto',
                                                        max_leaf_nodes=None,
                                                        min_impurity_decrease=0.0,
                                                        min_impurity_split=None,
                                                        min_samples_leaf=1,
                                                        min_samples_split=2,
                                                        min_weight_fraction_leaf=0.0,
                                                        n_estimators='warn',
                                                        n_jobs=None,
                                                        oob_s...
                       iid='warn', n_iter=100, n_jobs=-1,
                       param_distributions={'bootstrap': [True, False],
                                            'max_depth': [10, 20, 30, 40, 50, 60,
                                                          70, 80, 90, 100, 110,
                                                          None],
                                            'max_features': ['auto', 'sqrt'],
                                            'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10],
                                            'n_estimators': [100, 311, 522, 733,
                                                             944, 1155, 1366, 1577,
                                                             1788, 2000]},
                       pre_dispatch='2*n_jobs', random_state=1, refit=True,
                       return_train_score=False, scoring=None, verbose=2)




```python
#Running New RFC with New Hyperparameters

#Setting Up Features and Target Variables
target = df['NSP']
features = df.loc[:, df.columns != 'NSP']

#Train/Test Splitting with 80/20 Split
features_train2, features_test2, target_train2, target_test2 = train_test_split(features, target, test_size = 0.2, random_state = 1)

#Creating RFC Model
model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=1, verbose=0,
                       warm_start=False)

#Fitting Training Data
model.fit(features_train2, target_train2)

#Checking Predictions
rf_predictions = model.predict(features_test2)

#Generating Confusion Matrix
test2 = np.array(target_test2)
predictions2 = np.array(rf_predictions)
print(confusion_matrix(test2, predictions2))

#Classification Report
print(classification_report(test2, predictions2))
```

    [[326   0   0]
     [  2  66   0]
     [  0   1  31]]
                  precision    recall  f1-score   support

             1.0       0.99      1.00      1.00       326
             2.0       0.99      0.97      0.98        68
             3.0       1.00      0.97      0.98        32

        accuracy                           0.99       426
       macro avg       0.99      0.98      0.99       426
    weighted avg       0.99      0.99      0.99       426




```python
#Running KNN With Our Other Target Variables Eliminated A, B, C, D, E, AD, DE, LD, FS, SUSP, CLASS
df = df.drop(columns = ['A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP', 'CLASS'])
```


```python
#Setting New Features and Target and Checking KNN
target_new = df['NSP']
features_new = df.loc[:, df.columns != 'NSP']

#Getting Dummy Variables for our categorical variables
target_new = pd.get_dummies(target_new)
features_new = pd.get_dummies(features_new)

#Create Standardizer
standardizer = StandardScaler()

#Standardize Features
features_standardized = standardizer.fit_transform(features_new)

#Train/Test 80/20 Split
features_train, features_test, target_train, target_test = train_test_split(features_standardized, target_new, test_size = 0.2, random_state = 1)

#Creating Classifier
knn = KNeighborsClassifier(n_neighbors = 3, leaf_size = 1, p = 1, n_jobs = -1)

#Fitting Classifier on Trianing Data
knn.fit(features_train, target_train)

target_pred = knn.predict(features_test)
test0 = np.array(target_test).argmax(axis = 1)
predictions0 = np.array(target_pred).argmax(axis = 1)
print(confusion_matrix(test0, predictions0))

#Printing Classification Report for KNN
print(classification_report(test0, predictions0))
```

    [[319   6   1]
     [ 23  40   5]
     [  7   6  19]]
                  precision    recall  f1-score   support

               0       0.91      0.98      0.95       326
               1       0.77      0.59      0.67        68
               2       0.76      0.59      0.67        32

        accuracy                           0.89       426
       macro avg       0.81      0.72      0.76       426
    weighted avg       0.88      0.89      0.88       426




```python
#Hyperparameter Tuning for KNN Using Grid Search CV With the 10 Predictors Removed

#Creating Hyperparameter Grid
param_dist1 = {"leaf_size": list(range(1,50)),
              "n_neighbors": list(range(1,30)),
              "p": [1,2]}

#Create New KNN Object
knn_2 = KNeighborsClassifier()

#Use Gridsearch
clf = GridSearchCV(knn_2, param_dist1, cv=10, n_jobs=-1)

#Fit Model
best_model = clf.fit(features_standardized, target)

print('Best Leaf Size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
print('Best Metric:', best_model.best_estimator_.get_params()['metric'])
print('Best Weights:', best_model.best_estimator_.get_params()['weights'])
```

    Best Leaf Size: 1
    Best p: 2
    Best n_neighbors: 20
    Best Metric: minkowski
    Best Weights: uniform



```python
#Doing KNN Using Hyperparameters Found Above

#Setting New Features and Target and Checking KNN
target_new = df['NSP']
features_new = df.loc[:, df.columns != 'NSP']

#Getting Dummy Variables for our categorical variables
target_new = pd.get_dummies(target_new)
features_new = pd.get_dummies(features_new)

#Create Standardizer
standardizer = StandardScaler()

#Standardize Features
features_standardized = standardizer.fit_transform(features_new)

#Train/Test 80/20 Split
features_train, features_test, target_train, target_test = train_test_split(features_standardized, target_new, test_size = 0.2, random_state = 1)

#Creating Classifier
knn = KNeighborsClassifier(n_neighbors = 20, leaf_size = 1, p = 2, metric = "minkowski", weights = "uniform", n_jobs = -1)

#Fitting Classifier on Trianing Data
knn.fit(features_train, target_train)

target_pred = knn.predict(features_test)
test0 = np.array(target_test).argmax(axis = 1)
predictions0 = np.array(target_pred).argmax(axis = 1)
print(confusion_matrix(test0, predictions0))

#Printing Classification Report for KNN
print(classification_report(test0, predictions0))
```

    [[321   5   0]
     [ 32  35   1]
     [ 10   6  16]]
                  precision    recall  f1-score   support

               0       0.88      0.98      0.93       326
               1       0.76      0.51      0.61        68
               2       0.94      0.50      0.65        32

        accuracy                           0.87       426
       macro avg       0.86      0.67      0.73       426
    weighted avg       0.87      0.87      0.86       426



__Based on the results here, the model actually performed worse by the F1-score, even when using hyperparameter tuning, when removing the other 10-class predictors to be used as prediction for our target variable__


```python
#Fitting Data On Decision Tree Algorithm

#Setting Up Features and Target Variables
target_new_1 = df['NSP']
features_new_1 = df.loc[:, df.columns != 'NSP']

#Getting Dummy Variables for our categorical variables
target_new = pd.get_dummies(target_new_1)
features_new = pd.get_dummies(features_new_1)

#Train/Test Splitting with 80/20 Split
features_train11, features_test11, target_train11, target_test11 = train_test_split(features_new_1, target_new_1, test_size = 0.2, random_state = 1)

#Instantiating Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(features_train11, target_train11)

#Setting Target Prediction Variable Based on Features Test
target_pred1 = dt.predict(features_test11)

#Generating Confusion Matrix
test11 = np.array(target_test11)
predictions11 = np.array(target_pred1)
print(confusion_matrix(test11, predictions11))

#Classification Report for Decision Tree
print(classification_report(test11, predictions11))
```

    [[308  17   1]
     [ 16  46   6]
     [  1   2  29]]
                  precision    recall  f1-score   support

             1.0       0.95      0.94      0.95       326
             2.0       0.71      0.68      0.69        68
             3.0       0.81      0.91      0.85        32

        accuracy                           0.90       426
       macro avg       0.82      0.84      0.83       426
    weighted avg       0.90      0.90      0.90       426




```python
#Hyperparameter Tuning Using New Dataset

#Creating Hyperparameter Grid
param_dist = {"max_depth": [3, None],
             "criterion": ["gini", "entropy"]}

tree_cv = RandomizedSearchCV(dt, param_dist, cv = 10)

tree_cv.fit(features_train11, target_train11)

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_estimator_))
print("Best Score is {}".format(tree_cv.best_score_))
```

    C:\Users\blmay\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:266: UserWarning: The total space of parameters 4 is smaller than n_iter=10. Running 4 iterations. For exhaustive searches, use GridSearchCV.
      % (grid_size, self.n_iter, grid_size), UserWarning)


    Tuned Decision Tree Parameters: DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best')
    Best Score is 0.9335294117647058



```python
#Fitting Data On Decision Tree Algorithm With Hyperparameters Specified

#Setting Up Features and Target Variables
target_new_1 = df['NSP']
features_new_1 = df.loc[:, df.columns != 'NSP']

#Getting Dummy Variables for our categorical variables
target_new = pd.get_dummies(target_new_1)
features_new = pd.get_dummies(features_new_1)

#Train/Test Splitting with 80/20 Split
features_train11, features_test11, target_train11, target_test11 = train_test_split(features_new_1, target_new_1, test_size = 0.2, random_state = 1)

#Instantiating Decision Tree Classifier
dt = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
dt.fit(features_train11, target_train11)

#Setting Target Prediction Variable Based on Features Test
target_pred1 = dt.predict(features_test11)

#Generating Confusion Matrix
test11 = np.array(target_test11)
predictions11 = np.array(target_pred1)
print(confusion_matrix(test11, predictions11))

#Classification Report for Decision Tree
print(classification_report(test11, predictions11))
```

    [[314  10   2]
     [ 16  47   5]
     [  1   3  28]]
                  precision    recall  f1-score   support

             1.0       0.95      0.96      0.96       326
             2.0       0.78      0.69      0.73        68
             3.0       0.80      0.88      0.84        32

        accuracy                           0.91       426
       macro avg       0.84      0.84      0.84       426
    weighted avg       0.91      0.91      0.91       426



__Based on removing the 10 point classifier as predictors, the Decision Tree did perform worse though with the hyperparameter tuning that was used before, the model had fairly positive results except for the Suspect class with a lower F1 score as compared to Normal and Pathologic categories__


```python
#Running Random Forest Algorithm on New Dataset

#Setting Up Features and Target Variables
target_new_2 = df['NSP']
features_new_2 = df.loc[:, df.columns != 'NSP']

#Getting Dummy Variables for our categorical variables
target_new_2 = pd.get_dummies(target_new_2)
features_new_2 = pd.get_dummies(features_new_2)

#Train/Test Splitting with 80/20 Split
features_train22, features_test22, target_train22, target_test22 = train_test_split(features_new_2, target_new_2, test_size = 0.2, random_state = 1)

#Creating RFC Model
rf = RandomForestClassifier()

#Fitting Training Data
rf.fit(features_train22, target_train22)

#Checking Predictions
rf_predictions1 = rf.predict(features_test22)

#Generating Confusion Matrix
test22 = np.array(target_test22)
predictions22 = np.array(rf_predictions1)
print(confusion_matrix(test22.argmax(axis=1), predictions22.argmax(axis=1)))

#Classification Report
print(classification_report(test22, predictions22))
```

    [[325   0   1]
     [ 20  45   3]
     [  4   6  22]]
                  precision    recall  f1-score   support

               0       0.96      0.99      0.97       326
               1       0.88      0.66      0.76        68
               2       0.85      0.69      0.76        32

       micro avg       0.94      0.92      0.93       426
       macro avg       0.89      0.78      0.83       426
    weighted avg       0.94      0.92      0.92       426
     samples avg       0.92      0.92      0.92       426



    C:\Users\blmay\Anaconda3\lib\site-packages\sklearn\ensemble\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    C:\Users\blmay\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in samples with no predicted labels.
      'precision', 'predicted', average, warn_for)



```python
#Hyperparameter Tuning for Random Forest Algorithm with the New Dataset

#Number of tress in Random Forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]

#Number of Features to Consider At Each Split
max_features = ['auto', 'sqrt']

#Maximum Number of Levels in Tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

#Minimum Number of Samples Required At Each Leaf
min_samples_split = [2,5,10]

#Minimum Number of Samles Required at Each Leaf Node
min_samples_leaf = [1,2,4]

#Method of Selecting Samples for Training Each Tree
bootstrap = [True, False]

#Create Random Grid
random_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap}

#Using Random Grid to Search for Best Hyperparameters

rf_random1 = RandomForestClassifier()

rf_random2 = RandomizedSearchCV(estimator = rf_random1, param_distributions = random_grid, n_iter = 100, cv = 10, verbose = 2, random_state = 1, n_jobs = -1)

#Fitting the Model
rf_random2.fit(features_train22, target_train22)
```

    Fitting 10 folds for each of 100 candidates, totalling 1000 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  17 tasks      | elapsed:    6.4s
    [Parallel(n_jobs=-1)]: Done 138 tasks      | elapsed:   39.3s
    [Parallel(n_jobs=-1)]: Done 341 tasks      | elapsed:  1.8min
    [Parallel(n_jobs=-1)]: Done 624 tasks      | elapsed:  3.4min
    [Parallel(n_jobs=-1)]: Done 1000 out of 1000 | elapsed:  5.5min finished





    RandomizedSearchCV(cv=10, error_score='raise-deprecating',
                       estimator=RandomForestClassifier(bootstrap=True,
                                                        class_weight=None,
                                                        criterion='gini',
                                                        max_depth=None,
                                                        max_features='auto',
                                                        max_leaf_nodes=None,
                                                        min_impurity_decrease=0.0,
                                                        min_impurity_split=None,
                                                        min_samples_leaf=1,
                                                        min_samples_split=2,
                                                        min_weight_fraction_leaf=0.0,
                                                        n_estimators='warn',
                                                        n_jobs=None,
                                                        oob_s...
                       iid='warn', n_iter=100, n_jobs=-1,
                       param_distributions={'bootstrap': [True, False],
                                            'max_depth': [10, 20, 30, 40, 50, 60,
                                                          70, 80, 90, 100, 110,
                                                          None],
                                            'max_features': ['auto', 'sqrt'],
                                            'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10],
                                            'n_estimators': [100, 311, 522, 733,
                                                             944, 1155, 1366, 1577,
                                                             1788, 2000]},
                       pre_dispatch='2*n_jobs', random_state=1, refit=True,
                       return_train_score=False, scoring=None, verbose=2)




```python
#Running Random Forest Algorithm on New Dataset with Tuned Hyperparameters

#Setting Up Features and Target Variables
target_new_2 = df['NSP']
features_new_2 = df.loc[:, df.columns != 'NSP']

#Getting Dummy Variables for our categorical variables
target_new_2 = pd.get_dummies(target_new_2)
features_new_2 = pd.get_dummies(features_new_2)

#Train/Test Splitting with 80/20 Split
features_train22, features_test22, target_train22, target_test22 = train_test_split(features_new_2, target_new_2, test_size = 0.2, random_state = 1)

#Creating RFC Model
rf = RandomForestClassifier(bootstrap=True, class_weight='balanced',
                                                    criterion='gini',
                                                    max_depth=None,
                                                    max_features='auto',
                                                    max_leaf_nodes=None,
                                                    min_impurity_decrease=0.0,
                                                    min_impurity_split=None,
                                                    min_samples_leaf=1,
                                                    min_samples_split=2,
                                                    min_weight_fraction_leaf=0.0,
                                                    n_estimators=100,
                                                    n_jobs = -1)
#Fitting Training Data
rf.fit(features_train22, target_train22)

#Checking Predictions
rf_predictions1 = rf.predict(features_test22)

#Generating Confusion Matrix
test22 = np.array(target_test22)
predictions22 = np.array(rf_predictions1)
print(confusion_matrix(test22.argmax(axis=1), predictions22.argmax(axis=1)))

#Classification Report
print(classification_report(test22, predictions22))
```

    [[324   2   0]
     [ 23  42   3]
     [  5   1  26]]
                  precision    recall  f1-score   support

               0       0.94      0.99      0.97       326
               1       0.93      0.62      0.74        68
               2       0.90      0.81      0.85        32

       micro avg       0.94      0.92      0.93       426
       macro avg       0.92      0.81      0.85       426
    weighted avg       0.94      0.92      0.92       426
     samples avg       0.92      0.92      0.92       426



    C:\Users\blmay\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in samples with no predicted labels.
      'precision', 'predicted', average, warn_for)


__In this case as well, the Random Forest model did perform worse without the 10 classifier system that was in the model before.__

__Using the results above, both the Random Forest and KNN models performed similarly after hyperparameter tuning and removing the extraneous variables of e, LBE, and Median.  Further the 10 point classification variables used as predictors actually improved the F1, Precision, and Recall scores so those variables should be kept in the model.__


```python

```