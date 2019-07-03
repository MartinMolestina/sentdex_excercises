# Importing required libraries
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

# Calculating slope and constant of the linear regression with a function
def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b

# Calculatiing the squared error of data vs regression
def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig)**2)

# Calculating R-Squared as an error meassure
def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

# Creating random dataset to test regression
def create_dataset(length=40,variance=30,step=2,correlation=True):
    val = 1
    ys = []
    for _ in range(length):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [x for x in range(len(ys))]
    return np.array(xs, dtype=np.float64),np.array(ys,dtype=np.float64)


if __name__ == '__main__': 

    # Defining simple data for testing
    ## Commented since random datasets are created
    #xs = np.array([1,2,3,4,5], dtype=np.float64)
    #ys = np.array([5,4,6,5,6], dtype=np.float64)

    # Use function to create dataset
    # USER CAN MODIFY ARGS TO GET DIFFERENT RESULTS
    xs, ys = create_dataset(100,50,2,correlation='neg')

    # Running the function with the data
    m, b = best_fit_slope_and_intercept(xs,ys)

    # Printing the results for the user
    print(f'slope: {round(m, 3)} constant: {round(b, 3)}')

    # Creating a regression line for the data
    regression_line = [(m*x)+b for x in xs]

    # Predicting y for a value of x with the regression
    predict_x = random.randrange(0, len(xs))
    predict_y = (m*predict_x)+b
    print(f'For value of x={predict_x}, y={round(predict_y, 3)}')

    # Measuring the error with R-Squared
    r_squared = coefficient_of_determination(ys,regression_line)
    print(f'R-Squared: {round(r_squared, 4)}')

    # Set graphing style for better looking figures
    style.use('ggplot')
    # Plotting the results 
    plt.scatter(xs,ys,color='#003F72',label='data')
    plt.plot(xs, regression_line, label='regression line')
    plt.scatter(predict_x, predict_y, color='#00EE72', label='prediction')
    plt.legend(loc=4)
    plt.show()