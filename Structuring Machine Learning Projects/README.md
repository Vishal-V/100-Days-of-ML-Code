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

## Problem Statement 2
To help you practice strategies for machine learning, in this week we’ll present another scenario and ask how you would act. We think this “simulator” of working in a machine learning project will give a task of what leading a machine learning project could be like!
  
You are employed by a startup building self-driving cars. You are in charge of detecting road signs (stop sign, pedestrian crossing sign, construction ahead sign) and traffic signals (red and green lights) in images. The goal is to recognize which of these objects appear in each image. As an example, the above image contains a pedestrian crossing sign and red traffic lights
  
  
Your 100,000 labeled images are taken using the front-facing camera of your car. This is also the distribution of data you care most about doing well on. You think you might be able to get a much larger dataset off the internet, that could be helpful for training even if the distribution of internet data is not the same.