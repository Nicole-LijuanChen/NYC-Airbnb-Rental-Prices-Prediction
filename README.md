<!-- HEADER SECTION -->

<div class='header'> 
<!-- Your header image here -->
<div class='headingImage' id='mainHeaderImage' align="center">
    <img src="https://github.com/Nicole-LijuanChen/Patterns-in-Amazon-customer-ratings/blob/master/images/amazon.jpg" width='1200' height='500' ></img>
</div>

<!-- Put your badges here, either for fun or for information -->
<div align="center">
    <!-- Project Type -->
    <img src="https://img.shields.io/badge/Project Type-Machine Learning-purple?style=flat-square">
    <!-- Maintained? -->
    <img src="https://img.shields.io/badge/Maintained%3F-IN PROG-blue?style=flat-square"></img>
    <!-- License? (MIT is Standard, make sure you license your project via github) -->
    <img src="https://img.shields.io/github/license/boogiedev/automotive-eda?style=flat-square">
    <!-- Commit Activity? (Fill in the blanks) -->
    <img src="https://img.shields.io/github/commit-activity/m/your_username/your_repo_name?style=flat-square">
</div>

</br>

<!-- Brief Indented Explaination, you can choose what type you want -->
<!-- Type 1 -->
>NYC-Airbnb-Rental-Prices-Prediction



<!-- TABLE OF CONTENTS SECTION -->
<!-- 
In page linkings are kind of weird and follow a specific format, it can be done in both markdown or HTML but I am sticking to markdown for this one as it is more readable. 

Example:
- [Title of Section](#title-of-section)
  - [Title of Nested Section](#title-of-nested-section)

## Title of Section

### Title of Nested Section

When linking section titles with spaces in between, you must use a '-' (dash) to indicate a space, and the reference link in parentheses must be lowercase. Formatting the actual title itself has to be in markdown as well. I suggest using two hashtags '##' to emphasize it is a section, leaving the largest heading (single #) for the project title. With nested titles, just keep going down in heading size (###, ####, ...)
-->

## Table of Contents

<!-- Overview Section -->

- [Overview](#overview)
  - [Background & Motivation](#context)
  - [Purpose](#context)

<!-- Section 1 -->
- [Data PREP](#data)

<!-- Section 2 -->
- [Data Analysis](#EDA)

<!-- Section 3 -->
- [Feature engineering](#visualizations)

<!-- Section 4 -->
- [Create model](#visualizations)

<!-- Contributors -->
- [Interpret model](#contributors)

<!-- Credits -->
- [Future Steps](#credits)




<!-- Optional Line -->
---



## Overview

### Background & Motivation

Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. This dataset describes the listing activity and metrics in NYC, NY for 2019.

Source: kaggle.com


### Hypothesis
The hypothesis is that the average ratings in Books is different with Beauty, and the ratings in Books is higher than Beauty.

The first reason I set this assumption is that Amazon was originally engaged in book selling. But, it started sold Beauty goods in 2007. Amazon will be more experienced than Beauty in product selection, storage and deliver of books. Customers will have a better buying experience and give them higher ratings. Secondly,the quality of the book is relatively easy to measure. However.There is a personal preference for the quality of the Beauty product. For example, some people think the color of the same lipstick is too bright, while others think it is too dark.

H0: The mean ratings of these two category is the same 

Ha: The mean ratings of these two category is different
<!-- SECTION 1 -->
## Data PREP


I search the data from https://nijianmo.github.io/amazon/index.html. In Beaty cotegory, there are 5,269 customer ratings and reviews from 2007 to 2018. Books dataset includes 27,164,982 reviews from 1997 to 2018. 

Raw data:

<img src='https://github.com/Nicole-LijuanChen/Patterns-in-Amazon-customer-ratings/blob/master/images/row_data.png?raw=true'></img>

Scan the data to determine what I need. Using pyspark to clean data. Make the data readable.

<img src='https://github.com/Nicole-LijuanChen/Patterns-in-Amazon-customer-ratings/blob/master/images/data_sample_view.png?raw=true'></img>

I built a scraper to scrape ratings from the 27m + customer reviews. Then I create some functions to clean the data that are shored in eda-helper.py files.


The tech stack consists of Python 3, Numpy, Pandas,  Matplotlib, scraper

scripts in src/:

eda-helper.py a script which does initial data cleaning and feature engineering, turning the rdd collection into a numpy array or pandas dataframe.

Data in src/data/, the raw datasets: All_Beauty_5.json      Books_5.json 

<!-- SECTION 2 -->
## Data Analysis
According to the ratings count, more than 65% of customers scored a 5 on both products, but 91.8% customer rate Beauty at 5 star. It seems that customers are more satisfied with Beauty than Books. This requires further analysis of the data.

<img src='https://github.com/Nicole-LijuanChen/Patterns-in-Amazon-customer-ratings/blob/master/images/ratings_count.png?raw=true' width='800' height='auto'></img>

Calculate the average ratings by year, and compare the two categories. The data shows that the mean ratings of Beauty is greater than that of Books.

<img src='https://github.com/Nicole-LijuanChen/Patterns-in-Amazon-customer-ratings/blob/master/images/average_ratings_plot.png?raw=true'></img>

<!-- SECTION 3 -->
## Hypothesis Testing
#### Beauty VS Books
HO: The mean ratings of these two categories are the same 

H1: The mean ratings of these two categories are different

Alpha: 𝛼 = 0.01

Using Welch's t-test to calculate P-value

Ttest result:
p-value = 1.9895758749664697e-152

Conclusion:

p-value is small than alpha, there are 99% confidence to reject H0.
The mean ratings between Beauty and Books are different.

#### Beauty VS Books*1.10
When the ratings of Books times 1.10, Using Welch's t-test to calculate P-value again.

Ttest result:
pvalue=0.0009 (smaller than alpha 0.01 )

So I have 99% confidence to  conclude that the mean ratings of Beauty is 10% greater than that of Books.







## Final thoughts
The data analysis 

<img src='https://github.com/Nicole-LijuanChen/Patterns-in-Amazon-customer-ratings/blob/master/images/mean_std_plot.png?raw=true'></img>



## Future Steps
Next, I want to deep-dive the data,







<!-- Another line -->
---
