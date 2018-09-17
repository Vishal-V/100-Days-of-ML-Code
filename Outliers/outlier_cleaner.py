#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    error = []

    ### finds all the squared errors for values in the training set
    for i in range(0, len(ages)):
        value = pow((predictions[i]-net_worths[i]),2)
        error.insert(i, value)
        
    size = int(len(error) * 0.1)
    error.sort(reverse=True)   
    
    ### Remove the top 10% of data that has the most error
    thresh = error[size]
    
    ###Create the cleaned dataset
    index = 0
    for i in range(0, len(ages)):
        value = pow((predictions[i]-net_worths[i]),2)
        if(value <= thresh):
            cleaned_data.insert(index, [ages[i],net_worths[i], value])
            index += 1
    
    return cleaned_data

