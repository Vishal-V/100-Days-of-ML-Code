# Structuring Machine Learning Projects

## Problem Statement 1
This example is adapted from a real production application, but with details disguised to protect confidentiality.  
  
You are a famous researcher in the City of Peacetopia. The people of Peacetopia have a common characteristic: they are afraid of birds. To save them, you have to build an algorithm that will detect any bird flying over Peacetopia and alert the population.

The City Council gives you a dataset of 10,000,000 images of the sky above Peacetopia, taken from the city’s security cameras. They are labelled:  
  
y = 0:  There is no bird on the image  
y = 1:  There is a bird on the image  
Your goal is to build an algorithm able to classify new images taken by security cameras from Peacetopia.  

There are a lot of decisions to make:    
- What is the evaluation metric?
- How do you structure your data into train/dev/test sets?
- Metric of success

The City Council tells you that they want an algorithm that
- Has high accuracy
- Runs quickly and takes only a short time to classify a new image.
- Can fit in a small amount of memory, so that it can run in a small processor that the city will attach to many different security cameras.