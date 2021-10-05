# Predicting-Mortgage-Backed-Securities-Prepayment-Using-Machine-Learning
Predicting Mortgage Backed Securities Prepayment Using Machine Learning

App link - https://msbriskpredictor.herokuapp.com/

![MBS_image](https://user-images.githubusercontent.com/54249224/135236781-38eeced0-81b3-4abc-893d-cf1a049bc0a8.jpg)


## Introduction
Mortgage-backed securities (MBS) are securities backed by a collection of mortgage loans. While any type of mortgage loans, residential or commercial, can be used as a collateral
for a mortgage-backed security, most are backed by residential mortgages. A residential mortgage backed security (MBS), is a fixed income security, and is one of the largest asset
classes in the financial market. These securities are sold to investors. As the borrowers gradually pay off the underlying mortgage loans, the investors receive payments of
interest and principal. A large risk factor in MSB lies in the possibility of prepayments.Prepayments are payment by borrowers,who pay back a part, or the full amount of the loan
earlier than discussed in their residential mortgage contract.The various Machine Learning models that could predict the prepayment risk of residential mortgage loans by using
machine learning techniques like Logistic Regression and k-nearest neighbors (KNN) algorithms.

## Dataset
We use Freddie Mac's Single Family Loan-Level Dataset.This dataset contains approximately 3 million rows(morgages) originated from January 1,1999.This datset mainly consists of
field-rate single family mortgage loans with a maturity of 30 years.The dataset consists of two pieces:Original data and Performance data.The Original data contains static
information acquired at the time of loan origination,credit score,DTI ratio ,and LTV ratio are few examples.The performance data is reported monthly and includes dynamic information like loan age,and if the mortgage is prepaid in the current month.

## Important Features
![image](https://user-images.githubusercontent.com/54249224/135237293-c73558ad-1608-4e1d-965e-ba8e32be7df0.png)

*Credit Score:* - A number, prepared by third parties, summarizing the borrower’s creditworthiness, which may be indicative of the likelihood that the borrower will timely repay
future obligations.Generally, the credit score disclosed is the score known at the time of acquisition and is the score used to originate the mortgage.

*MORTGAGE INSURANCE PERCENTAGE (MIP %):* The percentage of loss coverage on the loan, at the time of Freddie Mac’s purchase of the mortgage loan that a mortgage insurer is
providing to cover losses incurred as a result of a default on the loan. 

*ORIGINAL COMBINED LOAN-TO-VALUE (OCLTV):* In the case of a purchase mortgage loan, the ratio is obtained by dividing the original mortgage loan amount on the note date plus any
secondary mortgage loan amount disclosed by the Seller by the lesser of the mortgaged property’s appraised value on the note date or its purchase price.

*DEBT-TO-INCOME (DTI) RATIO:* Disclosure of the debt to income ratio is based on (1) the sum of the borrower's monthly debt payments, including monthly housing expenses that
incorporate the mortgage payment the borrower is making at the time of the delivery of the mortgage loan to Freddie Mac, divided by (2) the total monthly income used to
underwrite the loan as of the date of the origination of the such loan.

*LOAN-TO-VALUE (LTV):* In the case of a purchase mortgage loan, the ratio obtained by dividing the original mortgage loan amount on the note date by the lesser of the mortgaged 
property’s appraised value on the note date or its purchase price.

*PREPAYMENT PENALTY MORTGAGE (PPM):* - Denotes whether the mortgage is a PPM. A PPM is a mortgage with respect to which the borrower is, or at any time has been, obligated to
pay a penalty in the event of certain repayments of principal.

*ORIGINAL LOAN TERM:* A calculation of the number of scheduled monthly payments of the mortgage based on the First Payment Date and Maturity Date.

*NUMBER OF BORROWERS:* The number of Borrower(s) who are obligated to repay the mortgage note secured by the mortgaged property.

## Exploratory Data Analysis(EDA)
The First task to perform on the dataset is EDA. Analyse and Cleaning the dataset,Exploring the features of dataset and drop the unimportant features from the dataset.

#### Credit Score Range:

![image](https://user-images.githubusercontent.com/54249224/135242556-86357b86-16a4-446a-a600-6c96d5a9f22c.png)

The Credit Score range is majorly lies between 500 to 800.

#### Analyzing the outliers using Boxplot:

![image](https://user-images.githubusercontent.com/54249224/135243612-9f5e5f99-220b-4566-8738-a7149963d75e.png)

#### Heatmap Ploting:
To understand the correlation of different Numeric features in the dataset.

![image](https://user-images.githubusercontent.com/54249224/135244951-1e14ebc9-7932-4224-9fa7-6c71e3549719.png)

#### Repay_Range vs LTV_range
Bar bar depicting the LTV based on range of repay in years

![image](https://user-images.githubusercontent.com/54249224/135245716-a5f6fc27-3409-4d67-a2bd-3ad783380b6e.png)

#### Target Variable
Our Target variable is EverDelinquent

![image](https://user-images.githubusercontent.com/54249224/135246281-00093d0c-c123-440c-8a7c-8c6c0cfb0c5a.png)

![image](https://user-images.githubusercontent.com/54249224/135259878-e87728e5-3f41-495d-8cf2-58a6511f5bb1.png)


The Dataset is Imblance , so first we need to be Balance the dataset.we use SMOTE tecnhique to Balance our dataset.

![image](https://user-images.githubusercontent.com/54249224/135261832-607a0872-7b00-422a-95e4-f28b3b378d6b.png)

Now, After Balancing the dataset

![image](https://user-images.githubusercontent.com/54249224/135246707-c5aded02-3322-48c3-b407-a74c56eb1bd0.png)

## Model Used:
This is a Classification problem so we use Logistic Regression and KNN for Model Building.
### Training and Testing dataset
Spliting our dataset into 80 % for Training and 20% for testing and take Random state value is 27.

![image](https://user-images.githubusercontent.com/54249224/135260402-2886ad4d-1a67-42ad-868b-e9e2e7c396b7.png)

### 1. Logistic Regression:
This machine leaning model commonly used for classification problems.it uses sigmoid function to transform dependent variable and assumes linear relationship between independent variable and the transform dependent variable.

![image](https://user-images.githubusercontent.com/54249224/135261962-4563782b-2b16-4b91-a02b-1a3242d64aa1.png)
![image](https://user-images.githubusercontent.com/54249224/135262027-42ead7b4-02e1-42fa-8c1b-445bd2552085.png)
![image](https://user-images.githubusercontent.com/54249224/135262100-ba497767-b923-4fad-abfa-0f2aba05e811.png)
![image](https://user-images.githubusercontent.com/54249224/135262289-35d5c521-5653-4d9c-821a-b90089b5f9ab.png)


### 2. k-nearest neighbors (KNN):
K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm.

Best value of K=1 for our dataset.
Parameter: n_neighbors=1
![image](https://user-images.githubusercontent.com/54249224/135261377-546b3752-17c8-41ba-aa4d-afa18b72218a.png)
![image](https://user-images.githubusercontent.com/54249224/135261422-923428c6-b551-445b-ac88-fbc48445a231.png)
![image](https://user-images.githubusercontent.com/54249224/135261473-26e3636d-c3e8-44d5-aae2-d292250ff2b0.png)

#### We acheive best accuracy in KNN model that is 83.37 %. Therefore we use this model for Deployment.

## Model Deployment
### We use Flask Framework to create web app for our model:
we created a python page as "app.py" in out deployment folder.

**STEPS for our web app development:**

**1.** Importing required Libraries and intialize the falsk object.

**2.** Load pickle file for KNN model

**3.** We have created our web page using **HTML** and styling of web page using **CSS**.

**4.** Now , we have to join our HTML and CSS page with web app using Flask **app.route('/')**.

**5.** Now we take all the Inputs.

**6.** After getting all the inputs finally predict the output using **app.route('/predict')**.

## Requirement:

![image](https://user-images.githubusercontent.com/54249224/135271252-fa05843b-2ba6-4039-a998-98d98375f950.png)


## User Inputs:

**.** CreditScore

**.** IsFirstTime

**.** DTI

**.** LTV

**.** MIP

**.** OCLTV

**.** OrigInterestRate

**.** OriginalLoanTerm

**.** MonthsRepayment

**.** PPM

**.** NumBorrowers

## Deploy Our Model Using Heroku Plateform:

**User Interface**

![image](https://user-images.githubusercontent.com/54249224/135269260-21efd2c8-5f1e-45b3-96ef-bf09a0d426ff.png)

**Output:**

![image](https://user-images.githubusercontent.com/54249224/135269700-a41ce4d1-13d8-43c4-b7fa-d8e1ec6cf37b.png)

**Conclusion:**

Above the output predicted by our model is **This user is good for loan** .
if Borrower is not good for loan than the model predict **This user is Risky to give loan**.

**Browse link:**

App link - https://msbriskpredictor.herokuapp.com/
