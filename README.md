# Transformations for Linear Models

### Use Cases
1. General Automated Feature Generation for Linear Models and Gradient Boosting Models (LightGBM, CatBoost, XGBoost)
1. Transformation of Higher-Dimensional Feature Space to Lower-Dimensional Feature Space.
1. Features Automatically Generated and Selected to Imitate the Performance of Non-linear models
1. Linear Models are Needed At Times When Latency Becomes An Important Concern

### How
1. Neural Network Identifies Important Features
1. Important Features 
1. Gradient Boosting Model Identifies Important Feature Interaction Pairs a


In data analysis transformation is the replacement of a variable by a function of that variable: for example, replacing a variable x by the square root of x or the logarithm of x. In a stronger sense, a transformation is a replacement that changes the shape of a distribution or relationship.

### Reasons

There are many reasons for transformation. In practice, a transformation often works, serendipitously, to do several of these at once, particularly to reduce skewness, to produce nearly equal spreads and to produce a nearly linear or additive relationship. But this is not guaranteed.

1. Convenience
A transformed scale may be as natural as the original scale and more convenient for a specific purpose (e.g.  percentages rather than original data, sines rather than degrees).

2. Reducing skewness
A transformation may be used to reduce skewness.  A distribution that is symmetric or nearly so is often easier to handle and interpret than a skewed distribution. More specifically, a normal or Gaussian distribution is often regarded as ideal as it is assumed by many statistical methods.To reduce right skewness, take roots or logarithms or reciprocals (roots are weakest). This is the commonest problem in practice. To reduce left skewness, take squares or cubes or higher powers.

3. Equal spreads

A transformation may be used to produce approximately equal spreads, despite marked variations in level, which again makes data easier to handle and interpret. Each data set or subset having about the same spread or variability is a condition called homoscedasticity: its opposite is called heteroscedasticity.  
    
4. Linear relationships

When looking at relationships between variables, it is often far easier to think about patterns that are approximately linear than about patterns that are highly curved. This is vitally important when using linear regression, which amounts to fitting such patterns to data.
    
5. Additive relationships

Relationships are often easier to analyse when additive rather than (say) multiplicative. Additivity is a vital issue in analysis of variance.
