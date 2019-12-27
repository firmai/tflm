# Transformations and Interactions for Linear Models 
With Feature Generation and Selection Techniques

The package is nice and simple and boils down to one command.

```python
import tflm
target = "target"
Train_lin_X,Train_lin_y, Test_lin_X, Test_lin_y = tflm.runner(Train, Test, Target)
```

Now just add it to your linear model

```python
from sklearn import linear_model
lm = linear_model.LinearRegression()
lm = lm.fit(Train_lin_X, Train_lin_y)
```
For now this only works with regression problems (continuous targets)

#### Install

```
pip install tflm
```

### Description

This advanced and automated model generates and selects feature tranformations and interactions using MLP neural networks with four single standing transformations (power, log, reciprocal and roots) and a Gradient Boosting Model with two interaction methods (multiplication and division) both using shapley additive contribution scores for selection criteria, the benefit of which is a selection of generated features that immitates neural networks and decision trees which is known to have synergetic ensembling properties. The final selection based on a validation set uses a the Least-angle regression (LARS) algorithm. 

The amount of feature can greatly inflate depending on the qulity of interactive effects and the benefits obtained from transformations. Although eight hyperparameters are made availble, for now I have chosen the parameters automatically based on data characteristics. These parameters in their current development state are fragile and on top of that hyperparameter selection is extremely expensive with this method. Each iteration passes through four selection filters. The data can pass multiple times through the tflm method, for now I have internally capped the iterations at two. This is an extremely slow algorithm, with the purpose of using a lot of time upfront to create good features that you can use a fast linear model in the future as opposed to a slow non-linear model. 

In data analysis, transformation is the replacement of a variable by a function of that variable: for example, replacing a variable x by the square root of x or the logarithm of x. In a stronger sense, a transformation is a replacement that changes the shape of a distribution or relationship. Interactions arise when considering the relationship among three or more variables. It describes a situation in which the effect of one variable on an outcome depends on the state of a second variable. Interaction terms can be created in various ways such as the product of x and y or the ratio of x and y. 



### Use Cases
1. General Automated Feature Generation for Linear Models and Gradient Boosting Models (LightGBM, CatBoost, XGBoost)
1. Transformation of Higher-Dimensional Feature Space to Lower-Dimensional Feature Space.
1. Features Automatically Generated and Selected to Imitate the Performance of Non-linear models
1. Linear Models are Needed at Times When Latency Becomes an Important Concern

### How
1. **MLP** Neural Network Identifies the Most Important Features for Interaction and Selection 
1. All Feature Importance and Feature Interaction Values are Based on **SHAP** (SHapley Additive exPlanations)
1. The Most Important Single Standing Features are Tranformed **POWER_2** (square) **LOG** (log plus 1) **RECIP** (reciprocal) **SQRT** (square root plus 1)
1. **GBM** Gradient Boosting Model uses the **MLP** Identified Important Features to Select a Subset of Important Interaction Pairs
1. The Most Important Interaction Pairs are Interacted **a_X_b** (multiplication) **c_DIV_h** (division) 
1. All Transformations are Fed as Input into an **MLP** model and Selected to **X%** (default 90%) Feature Contribution
1. The Whole Process is Repeated One More Time So That Higher Dimensional Interaction Can Take Place imagine **a_POWER_b_X_c_DIV_h**
1. Finally a **Lasso** Regression Selects Features from a Validation Set Using the **LARS algorithm** 

### To Do
1. Current parameter selection is based on data characteristics and bayesian hyperparameter optimisation could help.
1. Method for undoing interactions and transformations to identify original feature importance. 
1. Optimisation for users without access to GPUs (for now, you can use model="LightGBM" paramater).


### Example

Download Dataset and Activate Runner

```python
import tflm
import sklearn.datasets
from sklearn import linear_model

dataset = sklearn.datasets.fetch_california_housing()
X = pd.DataFrame(dataset['data'])
X["target"] = dataset["target"]
first = X.sample(int(len(X)/2))  # random selection leading to different scores
second = X[~X.isin(first)].dropna()
target = "target"

X_train, y_train, X_test, y_test = tflm.runner(first, second, target)
```
Modelling and MSE Score

```python

from sklearn import linear_model
lm = linear_model.LinearRegression()
lm = lm.fit(X_train,y_train)
preds = lm.predict(X_test)

mse = mean_squared_error(y_test, preds)
print(mse)
#Score Achieved = 0.43

```

Compare Performance With Untransformed Features

```python

import pandas as pd
from sklearn import preprocessing

def scaler(df):
  x = df.values #returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  df = pd.DataFrame(x_scaled)
  return df

add_first_y = first[target]
add_first = scaler(first.drop([target],axis=1))

add_second_y = second[target]
add_second = scaler(second.drop([target],axis=1)) 

from sklearn import linear_model
#clf = linear_model.Lasso(alpha=0.4)
clf = linear_model.LinearRegression()
preds = clf.fit(add_first,add_first_y).predict(add_second)
mse = mean_squared_error(add_second_y, preds)
print(mse)
#Score Achieved = 0.55
```
That is a performance improvement of more than 20% by using exactely the same data !!

That does not mean it always performs better than the standard data format; here is a Google Colab [example](https://colab.research.google.com/drive/1oEnsZ37FW266zdRK2Qa7del0T0ly-xKy) where this method performs poorly because of a lack of data. 



## Reasons

There are many reasons for transformation. In practice, a transformation often works, serendipitously, to do several of these at once, particularly to reduce skewness, to produce nearly equal spreads and to produce a nearly linear or additive relationship. But this is not guaranteed.

1. **Convenience**:
A transformed scale may be as natural as the original scale and more convenient for a specific purpose (e.g.  percentages rather than original data, sines rather than degrees).

2. **Reducing skewness**:
A transformation may be used to reduce skewness.  A distribution that is symmetric or nearly so is often easier to handle and interpret than a skewed distribution. More specifically, a normal or Gaussian distribution is often regarded as ideal as it is assumed by many statistical methods.To reduce right skewness, take roots or logarithms or reciprocals (roots are weakest). This is the commonest problem in practice. To reduce left skewness, take squares or cubes or higher powers.

3. **Equal spreads**:
A transformation may be used to produce approximately equal spreads, despite marked variations in level, which again makes data easier to handle and interpret. Each data set or subset having about the same spread or variability is a condition called homoscedasticity: its opposite is called heteroscedasticity.  
    
4. **Linear relationships**:
When looking at relationships between variables, it is often far easier to think about patterns that are approximately linear than about patterns that are highly curved. This is vitally important when using linear regression, which amounts to fitting such patterns to data.
    
5. **Additive relationships**
Relationships are often easier to analyse when additive rather than (say) multiplicative. Additivity is a vital issue in analysis of variance.

## Transformations Implemented



The most useful transformations in introductory data analysis are the
    reciprocal, logarithm, cube root, square root, and square. In what
    follows, even when it is not emphasised, it is supposed that
    transformations are used only over ranges on which they yield (finite)
    real numbers as results.


    Reciprocal


    The reciprocal, x to 1/x, with its sibling the negative reciprocal, x to
    -1/x, is a very strong transformation with a drastic effect on
    distribution shape. It can not be applied to zero values.  Although it
    can be applied to negative values, it is not useful unless all values are
    positive. The reciprocal of a ratio may often be interpreted as easily as
    the ratio itself: e.g.


        population density (people per unit area) becomes area per person;


        persons per doctor becomes doctors per person;


        rates of erosion become time to erode a unit depth.


    (In practice, we might want to multiply or divide the results of taking
    the reciprocal by some constant, such as 1000 or 10000, to get numbers
    that are easy to manage, but that itself has no effect on skewness or
    linearity.)


    The reciprocal reverses order among values of the same sign: largest
    becomes smallest, etc. The negative reciprocal preserves order among
    values of the same sign.


    Logarithm


    The logarithm, x to log base 10 of x, or x to log base e of x (ln x), or
    x to log base 2 of x, is a strong transformation with a major effect on
    distribution shape. It is commonly used for reducing right skewness and
    is often appropriate for measured variables. It can not be applied to
    zero or negative values. One unit on a logarithmic scale means a
    multiplication by the base of logarithms being used. Exponential growth
    or decline


        y = a exp(bx)


    is made linear by


        ln y = ln a + bx


    so that the response variable y should be logged. (Here exp() means
    raising to the power e, approximately 2.71828, that is the base of
    natural logarithms.)


    An aside on this exponential growth or decline equation: put x = 0, and


        y = a exp(0) = a,


    so that a is the amount or count when x = 0. If a and b > 0, then y grows
    at a faster and faster rate (e.g. compound interest or unchecked
    population growth), whereas if a > 0 and b < 0, y declines at a slower
    and slower rate (e.g. radioactive decay).


    Power functions y = ax^b are made linear by log y = log a + b log x so
    that both variables y and x should be logged.


    An aside on such power functions: put x = 0, and for b > 0,


        y = ax^b = 0,
                 
    so the power function for positive b goes through the origin, which often
    makes physical or biological or economic sense. Think: does zero for x
    imply zero for y? This kind of power function is a shape that fits many
    data sets rather well.


    Consider ratios y = p / q where p and q are both positive in practice.
    Examples are


        males / females;
        dependants / workers;
        downstream length / downvalley length. 


    Then y is somewhere between 0 and infinity, or in the last case, between
    1 and infinity. If p = q, then y = 1. Such definitions often lead to
    skewed data, because there is a clear lower limit and no clear upper
    limit. The logarithm, however, namely


        log y = log p / q = log p - log q,


    is somewhere between -infinity and infinity and p = q means that log y =
    0. Hence the logarithm of such a ratio is likely to be more symmetrically
    distributed.


    Cube root 


    The cube root, x to x^(1/3). This is a fairly strong transformation with
    a substantial effect on distribution shape: it is weaker than the
    logarithm. It is also used for reducing right skewness, and has the
    advantage that it can be applied to zero and negative values.  Note that
    the cube root of a volume has the units of a length. It is commonly
    applied to rainfall data.


    Applicability to negative values requires a special note. Consider
    (2)(2)(2) = 8 and (-2)(-2)(-2) = -8. These examples show that the cube
    root of a negative number has negative sign and the same absolute value
    as the cube root of the equivalent positive number. A similar property is
    possessed by any other root whose power is the reciprocal of an odd
    positive integer (powers 1/3, 1/5, 1/7, etc.).


    This property is a little delicate. For example, change the power just a
    smidgen from 1/3, and we can no longer define the result as a product of
    precisely three terms. However, the property is there to be exploited if
    useful.


    Square root


    The square root, x to x^(1/2) = sqrt(x), is a transformation with a
    moderate effect on distribution shape: it is weaker than the logarithm
    and the cube root. It is also used for reducing right skewness, and also
    has the advantage that it can be applied to zero values. Note that the
    square root of an area has the units of a length. It is commonly applied
    to counted data, especially if the values are mostly rather small.


    Square 


    The square, x to x^2, has a moderate effect on distribution shape and it
    could be used to reduce left skewness. In practice, the main reason for
    using it is to fit a response by a quadratic function y = a + b x + c
    x^2. Quadratics have a turning point, either a maximum or a minimum,
    although the turning point in a function fitted to data might be far
    beyond the limits of the observations. The distance of a body from an
    origin is a quadratic if that body is moving under constant acceleration,
    which gives a very clear physical justification for using a quadratic.
    Otherwise quadratics are typically used solely because they can mimic a
    relationship within the data region. Outside that region they may behave
    very poorly, because they take on arbitrarily large values for extreme
    values of x, and unless the intercept a is constrained to be 0, they may
    behave unrealistically close to the origin.


    Squaring usually makes sense only if the variable concerned is zero or
    positive, given that (-x)^2 and x^2 are identical.


Additional information on tranformations, and a blog post partly inspiring the use of transformations in this package and the content in this readm can be found [here](http://fmwww.bc.edu/repec/bocode/t/transint.html). 
