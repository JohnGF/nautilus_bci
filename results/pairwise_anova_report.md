# BCI Music Pairwise Classification & ANOVA Report
Analyzed Sessions: `['01']` | Common Channels: `21` | Total Trials: `30`

## 1. Pairwise Binary Decoding Rankings (Chance: 50.0%)
Evaluates which song recalls elicit the most distinct EEG spatial patterns. Higher accuracy means the two songs are **easier to differentiate** in your brain.

| Rank | Song A | Song B | Binary BCI Accuracy | Classification Margin |
| :---: | :--- | :--- | :---: | :---: |
| 1 | Track_2 | Track_6 | **80.0%** | +30.0% |
| 2 | Track_3 | Track_6 | **70.0%** | +20.0% |
| 3 | Track_2 | Track_3 | **60.0%** | +10.0% |
| 4 | Track_3 | Track_5 | **60.0%** | +10.0% |
| 5 | Track_4 | Track_6 | **60.0%** | +10.0% |
| 6 | Track_2 | Track_4 | **50.0%** | +0.0% |
| 7 | Track_2 | Track_5 | **50.0%** | +0.0% |
| 8 | Track_3 | Track_4 | **50.0%** | +0.0% |
| 9 | Track_5 | Track_6 | **50.0%** | +0.0% |
| 10 | Track_1 | Track_2 | **40.0%** | -10.0% |
| 11 | Track_1 | Track_3 | **40.0%** | -10.0% |
| 12 | Track_1 | Track_4 | **40.0%** | -10.0% |
| 13 | Track_1 | Track_5 | **30.0%** | -20.0% |
| 14 | Track_1 | Track_6 | **30.0%** | -20.0% |
| 15 | Track_4 | Track_5 | **30.0%** | -20.0% |

## 2. Multi-Class EEG Band Power ANOVA Statistics
Determines if there is a statistically significant difference in global brain rhythm power levels across the 6 different songs. A **p-value < 0.05** represents statistical significance.

| Brain Rhythm | F-Statistic | P-Value | Statistical Significance |
| :--- | :---: | :---: | :--- |
| Theta (4-8 Hz) | 0.722 | 0.61378 | ❌ NOT SIGNIFICANT |
| Alpha (8-12 Hz) | 0.751 | 0.59335 | ❌ NOT SIGNIFICANT |
| Beta (13-30 Hz) | 0.785 | 0.57064 | ❌ NOT SIGNIFICANT |
