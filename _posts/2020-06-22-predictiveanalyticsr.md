---
title: "Using ML to Determine Infant Heart Rate Tracings - R Portion (Part 1)"
date: 2020-06-22
classes: wide
header:
  image: "/images/datamining/a-screen-showing-an-echocardiogram.jpg"
excerpt: "Predictive Analytics, Exploratory Data Analysis"
---

# Problem Background:

This project is divided into two parts.  The first part is this R-based portion for exploratory data analysis.  The second part is a Python based portion in another post to utilize machine learning algorithms.

Fetal heart rate monitoring as an indicator of fetal well-being can be inaccurate as predictors of a poor neonatal outcome and come with significant healthcare and medicolegal costs.  It is the goal of this project to use a database of specific technical characteristics of fetal heart rate monitoring from the UCI machine learning database to develop a predictive model using an automated system to better identify worrisome decreases in fetal heart rate.

The data that I will primarily use is from the University of California – Irvine Machine Learning Repository.  The dataset can be found at the following web address: [https://archive.ics.uci.edu/ml/datasets/Cardiotocography](https://archive.ics.uci.edu/ml/datasets/Cardiotocography).

According to the website, there were over 2000 fetal heart tracings (cardiotocograms) and interpreted by three expert obstetricians.  Many of the measurements include the technical measurements include heart rate accelerations, decelerations, max heart rate, minimum heart rates, heart rate baseline and finally the target variable is whether the tracing was normal, suspect, or pathologic.

There were two different types of target variables included in this dataset.  There were 10 different categorical variables that when put together could be interpreted as suspect or not.  The other target variable was a three class categorical variable describing normal, suspect, or pathologic.

The purpose of this portion is for data exploration and analysis and insight for further model fitting and prediction.

## Initial Variable Description of Dataset

__FileName and SegFile__ = Personal Identifiers

__Date__ = Individual Date of Measurement

__b__ = Starting Point of Measurement

__e__ = Ending Point of Measurement

__LBE__ = Baseline value (By Medical Expert)

__LB__ = Fetal Heart Rate Baseline (beats/minute - Automated)

__AC__ = # Accelerations/second

__FM__ = # Fetal Movements/second

__UC__ = # Uterine Contractions/second

__DL__ = # Light Decelerations/second

__DS__ = # Severe Decelerations/second

__DP__ = # Prolonged Decelerations/second

__DR__ = # Repetitive Decelerations (All 0s)

__ASTV__ = % of time with abnormal short term variability

__MSTV__ = Mean Value of Short Term Variability

__ALTV__ = % of time with abnormal long term variability

__MLTV__ = Mean Value of Long Term Variability

__Width__ = Width of Fetal Heart Rate Histogram

__Min__ = Minimum of Fetal Heart Rate Histogram

__Max__ = Maximum of Fetal Heart Rate Histogram

__Nmax__ = # Histogram Peaks

__Nzeros__ = # Histogram Zeros

__Mode__ = Histogram Mode

__Mean__ = Histogram Mean

__Median__ = Histogram Median

__Variance__ = Histogram Variance

__Tendency__ = Histogram Tendency:  -1 = Left Asymmetric, 0 = Symmetric, 1 = Right Asymmetric

__A__ = Calm Sleep

__B__ = REM Sleep

__C__ = Calm Vigilance

__D__ = Active Vigilance

__AD__ = Accelerative/Decelerative Pattern (Stress Situation)

__DE__ = Decelerative Pattern (Vagal Stimulation)

__LD__ = Largely decelerative pattern

__FS__ = Flat Sinusoidal Pattern (Pathologic State)

__SUSP__ = Suspect Pattern

__Class__ = 1 to 10 for Classes A to SUSP (Target Based on the 10 Preceding Factors)

__NSP__ = Fetal State Class Code (Normal = 1, Suspect = 2, Pathologic = 3)

At the end of the spreadsheet, there were four rows that had null values and we will remove those including the identifier and date columns as they are not needed.

```r

df <- toco %>% select(-FileName, -Date, -SegFile)
df <- na.omit(df)
summary(df)
```

The categorical variables (Tendency, A, B, C, D, AD, DE, LD, FS, SUSP, CLASS, NSP) were recoded in R to ensure they were being handled correctly for descriptive analysis.

```r
cols = (c("Tendency", "A", "B", "C", "D", "AD", "DE", "LD", "FS", "SUSP", "CLASS", "NSP"))

df[cols] <- lapply(df[cols], factor)
df$E <- as.factor(df$E)
str(df)
```

Next, histograms and bar graphs were plotted to check the distributions of the variables.

```r
hist(df$LBE, main = "Histogram of Baseline Values (Expert)", xlab = "Baseline HR", ylab = "Counts")
hist(df$LB, main = "Histogram of Baseline Values (Automated)", xlab = "Baseline HR", ylab = "Counts")
hist(df$AC, main = "Histogram of Accelerations", xlab = "Accelerations", ylab = "Counts")
hist(df$FM, main = "Histogram of Fetal Movement", xlab = "Movements", ylab = "Counts")
hist(df$ASTV, main = "Histogram of Percentage of Time With Abnormal Short-Term Variability", xlab = "Percentage", ylab = "Counts")
hist(df$MSTV, main = "Histogram of Mean Time With Abnormal Short-Term Variability", ylab = "Counts")
hist(df$ALTV, main = "Histogram of Percentage of Time With Abnormal Long-Term Variability", xlab = "Percentage", ylab = "Counts")
hist(df$MLTV, main = "Histogram of Mean Time With Abnormal Long-Term Variability", ylab = "Counts")
hist(df$DL, main = "Histogram of Light Decelerations", xlab = "Number of Decels", ylab = "Counts")
hist(df$DS, main = "Histogram of Severe Decelerations", xlab = "Number of Decels", ylab = "Counts")
hist(df$DP, main = "Histogram of Prolonged Decelerations", xlab = "Number of Decels", ylab = "Counts")
hist(df$DR, main = "Histogram of Repeated Decelerations", xlab = "Number of Decels", ylab = "Counts")
hist(df$Width, main = "Histogram Width", ylab = "Counts")
hist(df$Min, main = "Histogram of Minimum HR", ylab = "Counts")
hist(df$Max, main = "Histogram of Maximum HR", ylab = "Counts")
hist(df$Nmax, main = "Number of Histogram Peaks", ylab = "Counts")
hist(df$Nzeros, main = "Number of Histogram Zeros", ylab = "Counts")
hist(df$Mode, main = "Histogram Mode", ylab = "Counts")
hist(df$Median, main = "Histogram Median", ylab = "Counts")
hist(df$Variance, main = "Histogram Variance", ylab = "Counts")
plot(df$Tendency, main = "Distribution of Histogram Tendencies", ylab = "Counts")
plot(df$A, main = "Distribution of Calm Sleep", ylab = "Counts")
plot(df$B, main = "Distribution of REM Sleep", ylab = "Counts")
plot(df$C, main = "Distribution of Calm Vigilance", ylab = "Counts")
plot(df$D, main = "Distribution of Active Vigilance", ylab = "Counts")
plot(df$AD, main = "Distribution of Acceleration/Deceleration Pattern (Stress)", ylab = "Counts")
plot(df$DE, main = "Distribution of Deceleration (Vagal Stimulation)", ylab = "Counts")
plot(df$LD, main = "Distribution of Largely Decelerative Pattern", ylab = "Counts")
plot(df$FS, main = "Distribution of Flat Sinusoidal Pattern", ylab = "Counts")
plot(df$SUSP, main = "Distribution of Suspect Pattern", ylab = "Counts")
plot(df$CLASS, main = "Distribution of Class Code (A to SUSP)", ylab = "Counts")
plot(df$NSP, main = "Distribution of Overall Classification", ylab = "Counts")
```

## Initial Analysis:

Both baseline FHR by both expert and computerized measurements appeared normally distributed with an approximately normal range of FHR 110 - 160.  This is the expected normal range of a fetal heart rate during labor and delivery.  Any heart rates outside these ranges is abnormal and requires investigation and/or intervention.  There was not a significant difference between the two values.

![png](/images/predictiveanalytics/000003.png)

![png](/images/predictiveanalytics/000005.png)

After visualizing the distribution of the target variable, the vast majority were interpreted as Normal (1), with the next most frequent being Suspect (2) or potentially abnormal, and the fewest being interpreted as Pathologic (3).  This can affect the choice of our success metrics after model fitting and prediction.

![png](/images/predictiveanalytics/000041.png)

As can be seen above, the vast majority of the records were read as normal.  However, for prediction, the algorithm will have to be fine-tuned to weigh the target classes appropriately.  If using accuracy solely as a metric, the algorithm would be more likely to be right if it always assumed that the tracing was normal.

Several of the variables appeared positively skewed meaning that there were a larger number of values at the lower end of the x values.  These variables were: Accelerations, Fetal Movement, Mean Time With Abnormal Short-Term Variability, % of time with abnormal long-term variability, mean time with abnormal long-term variability, Light Decelerations, Severe Decelerations, Prolonged Decelerations, and Number of Histogram Peaks and Zeros.  Several are included below.

![png](/images/predictiveanalytics/000007.png)

![png](/images/predictiveanalytics/000009.png)

![png](/images/predictiveanalytics/000011.png)

![png](/images/predictiveanalytics/000013.png)

![png](/images/predictiveanalytics/000017.png)

One finding that I think bears mentioning is that the variable for repeated decelerations were all 0's meaning there were no cases with repetitive decelerations, which could be concerning.  

Further, looking at some of the 10 point categorical variables, the vast majority of the tracings did not indicate that condition.  For example, there were significantly less acceleration/deceleration patterns indicative of stress, vagal stimulation, and a largely decelerative pattern.  Some of these could be indicative of fetal distress.

![png](/images/predictiveanalytics/000035.png)

![png](/images/predictiveanalytics/000037.png)

![png](/images/predictiveanalytics/000039.png)

A flat sinusoidal pattern is highly suspicious for fetal distress and hypoxia.  Likewise, very few of the tracings in the dataset had this concerning finding.

![png](/images/predictiveanalytics/00003b.png)

Next, we will plan on exploring some bivariate plots to analyze relationships.  One that I am most interested in is the relation between fetal heart rate and the target variable of normal, suspect, or pathologic.

```r
ggplot(df, aes(as.factor(df$NSP), df$b)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Starting Measurement Point') + ggtitle('Scatterplot of Tracing Classification and Beginning Measurement')
ggplot(df, aes(as.factor(df$NSP), df$e)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Ending Measurement Point') + ggtitle('Scatterplot of Tracing Classification and Ending Measurement')
ggplot(df, aes(as.factor(df$NSP), df$LBE)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Mean Fetal Heart Rate') + ggtitle('Scatterplot of Tracing Classification and Fetal Heart Rate (Expert)')
ggplot(df, aes(as.factor(df$NSP), df$LB)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Mean Fetal Heart Rate') + ggtitle('Scatterplot of Tracing Classification and Fetal Heart Rate (Automated)')
ggplot(df, aes(as.factor(df$NSP), df$AC)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Number of Accelerations/Second') + ggtitle('Scatterplot of Tracing Classification and Number of Accelerations/Second')
ggplot(df, aes(as.factor(df$NSP), df$FM)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Number of Fetal Movements/Second') + ggtitle('Scatterplot of Tracing Classification and Number of Fetal Movements/Second')
ggplot(df, aes(as.factor(df$NSP), df$UC)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Number of Uterine Contractions/Second') + ggtitle('Scatterplot of Tracing Classification and Number of Uterine Contractions/Second')
ggplot(df, aes(as.factor(df$NSP), df$DL)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Number of Light Decelerations/Second') + ggtitle('Scatterplot of Tracing Classification and Number of Light Decelerations/Second')
ggplot(df, aes(as.factor(df$NSP), df$DS)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Number of Severe Decelerations/Second') + ggtitle('Scatterplot of Tracing Classification and Number of Severe Decelerations/Second')
ggplot(df, aes(as.factor(df$NSP), df$DP)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Number of Prolonged Decelerations/Second') + ggtitle('Scatterplot of Tracing Classification and Number of Prolonged Decelerations/Second')
ggplot(df, aes(as.factor(df$NSP), df$ASTV)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Number of % of Time with Abnormal Short-Term Variability') + ggtitle('Scatterplot of Tracing Classification and % of Time with Abnml. Short-Term Variability')
ggplot(df, aes(as.factor(df$NSP), df$MSTV)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Mean Value of Short-Term Variability') + ggtitle('Scatterplot of Tracing Classification and Mean Value of Short-Term Variability')
ggplot(df, aes(as.factor(df$NSP), df$ALTV)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('% of Time with Abnormal Long-Term Variability') + ggtitle('Scatterplot of Tracing Classification and % of Time with Abnml. Long-Term Varaibility')
ggplot(df, aes(as.factor(df$NSP), df$MLTV)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Mean Value of Long-Term Variability') + ggtitle('Scatterplot of Tracing Classification and Mean Value of Long-Term Variability')
ggplot(df, aes(as.factor(df$NSP), df$Width)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Histogram Width') + ggtitle('Scatterplot of Tracing Classification and Histogram Width')
ggplot(df, aes(as.factor(df$NSP), df$Min)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Histogram Minimum') + ggtitle('Scatterplot of Tracing Classification and Histogram Minimum Value')
ggplot(df, aes(as.factor(df$NSP), df$Max)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Histogram Maximum') + ggtitle('Scatterplot of Tracing Classification and Histogram Maximum')
ggplot(df, aes(as.factor(df$NSP), df$Nmax)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Number of Histogram Peaks') + ggtitle('Scatterplot of Tracing Classification and Number of Histogram Peaks')
ggplot(df, aes(as.factor(df$NSP), df$Nzeros)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Number of Histogram Zeros') + ggtitle('Scatterplot of Tracing Classification and Number of Histogram Zeros')
ggplot(df, aes(as.factor(df$NSP), df$Mode)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Fetal Heart Rate Histogram Mode') + ggtitle('Scatterplot of Tracing Classification and Fetal Heart Rate Histogram Mode')
ggplot(df, aes(as.factor(df$NSP), df$Mean)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Fetal Heart Rate Histogram Mean') + ggtitle('Scatterplot of Tracing Classification and Fetal Heart Rate Histogram Mean')
ggplot(df, aes(as.factor(df$NSP), df$Median)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Fetal Heart Rate Histogram Median') + ggtitle('Scatterplot of Tracing Classification and Fetal Heart Rate Histogram Median')
ggplot(df, aes(as.factor(df$NSP), df$Variance)) + geom_boxplot() + xlab('Classification (1-Normal, 2- Suspect, 3-Pathologic)') + ylab('Fetal Heart Rate Histogram Variance') + ggtitle('Scatterplot of Tracing Classification and Fetal Heart Rate Histogram Variance')
```

Looking at the relationship between the expert fetal heart rate determination and the classification of the fetal heart rate, there were outliers in the pathologic category that tended to be associated with higher values.  Further, in the suspect category, the fetal heart rates tended to have higher mean heart rates.

![png](/images/predictiveanalytics/000035 (1).png)

On exploratory analysis of the input variables to the target variable of NSP, there were several trends that are consistent with known associations of pathologic categories.  There was higher variance of values in the histograms of the fetal heart rates in the pathologic category.  There also tended to be higher amounts of time with percentage of time spent with abnormal short and long-term variability.  The number of uterine contractions/second were lower in the pathologic category.  The number of decelerations were higher in the pathologic category and FHR tended to be lower in the pathologic categories as well.  This validates the common findings that more pathologic findings are noted in fetal heart rate tracings with less variability, lower fetal heart rates, and more decelerations.

![png](/images/predictiveanalytics/00005d.png)

![png](/images/predictiveanalytics/000045.png)

![png](/images/predictiveanalytics/00003b (1).png)

![png](/images/predictiveanalytics/000043.png)

The data was examined for high levels of correlation between the input variables to look for redundancy.  The categorical variables (almost all of which are the target variables) were excluded.

```r
library(corrplot)

nums <- df %>% select(-A, -B, -C, -D, -E, -AD, -DE, -LD, -FS, -SUSP, -CLASS, -NSP, -Tendency, -DR)
corr <- cor(nums)
corr
corrplot(corr, method="circle")
```

A correlation plot was created to look for high levels of correlation between the input variables and only the numerical variables were included.  There appeared to be high levels of correlation between the beginning and ending measurements variable in addition to high levels of correlation between expert and automated determination of fetal heart rate.  Finally, mode, mean, and median variables were all highly correlated with each other based on the histogram values.  These variables will need to be assessed using feature selection to see if simplification will increase predictive accuracy.

Please continued to the next portion of the project using Python.
