# import system libraries
import numpy as np
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import weibull_min
from scipy.stats import beta
from scipy.stats import burr
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import csv

# params gives the shape of the distribution which is [a,b,loc,scale]
# Fit data to a normal distribution
def fit_normal(data):
    # fit data to a normal distribution
    mu, std = norm.fit(data)
    normal_parameters = norm.fit(data)

    # define a horizontal axis for plotting pdf and cdf
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    # solve distribution pdf
    pdf = norm.pdf(x, mu, std)

    # solve distribution cdf
    cdf = norm.cdf(x, mu, std)

    # Check tolerance interval.
    # For 95% upper limit use 90% in the function.
    # This is a two-sided function.
    norm_interval = stats.norm.interval(0.90, loc=mu, scale=std)

    # Check p value
    distribution_norm = stats.norm
    d_norm, pvalue_norm = stats.kstest(data, distribution_norm.cdf, normal_parameters)
    return norm_interval, pvalue_norm, normal_parameters, pdf, cdf

# Fit data to a lognormal distribution
def fit_lognormal(data):
    # lognormal parameters are shape, location and scale correspondingly.
    # log(scale) gives the mean
    # shape gives the std
    lognormal_parameters = stats.lognorm.fit(data, floc=0)

    # solve lognormal distribution values (For debugging purpose)
    lognormal_values = stats.lognorm.stats(*lognormal_parameters)

    # define a horizontal axis for plotting pdf and cdf
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    # solve distribution pdf
    pdf = stats.lognorm.pdf(x, *lognormal_parameters)

    # solve distribution cdf
    cdf = stats.lognorm.cdf(x, *lognormal_parameters)

    # Calculate tolerance interval
    lognormal_interval = stats.lognorm.interval(0.90, *lognormal_parameters)

    # Calculate p-value with KS test
    distribution_log = stats.lognorm
    d_lognorm, pvalue_lognorm = stats.kstest(data, distribution_log.cdf, lognormal_parameters)
    return lognormal_interval, pvalue_lognorm, lognormal_parameters, pdf, cdf

# Fit data to a weibull distribution
def fit_weibull(data):
    # fit data to a weibull distribution
    weib_params = stats.weibull_min.fit(data, floc=0, f0=1)

    # solve weibull distribution values
    weib_values = stats.weibull_min.stats(*weib_params)

    # define a horizontal axis for plotting pdf and cdf
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    # solve distribution pdf
    pdf = stats.weibull_min.pdf(x, *weib_params)

    # solve distribution pdf
    cdf = stats.weibull_min.cdf(x, *weib_params)

    # Check tolerance interval
    weib_interval = stats.weibull_min.interval(0.90, *weib_params)

    # Calculate P-value with KS test
    distribution_weibmin = stats.weibull_min
    d_weibmin, pvalue_weibmin = stats.kstest(data, distribution_weibmin.cdf, weib_params)
    return weib_interval, pvalue_weibmin, weib_params, pdf, cdf

# fit data to a beta distribution
def fit_beta(data):
    # fit to a beta distribution
    beta_parameters = stats.beta.fit(data, floc=0)

    # solve beta distribution params
    # beta_values = stats.lognorm.stats(*beta_parameters)

    # define a horizontal axis for plotting pdf and cdf
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    # solve pdf of the distribution
    pdf = stats.beta.pdf(x, *beta_parameters)

    # solve cdf of the distribution
    cdf = stats.beta.cdf(x, *beta_parameters)

    # Calculate tolerance interval
    beta_interval = stats.beta.interval(0.90, *beta_parameters)

    # Calculate p-value with KS test
    distribution_beta = stats.beta
    d_beta, pvalue_beta = stats.kstest(data, distribution_beta.cdf, beta_parameters)
    return beta_interval, pvalue_beta, beta_parameters, pdf, cdf

# fit data to a burr distribution
def fit_burr(data):
    # Fit data to a burr distribution
    burr_parameters = stats.burr.fit(data, floc=0)

    # Solve distribution values
    burr_values = stats.burr.stats(*burr_parameters)

    # define a horizontal axis for plotting pdf and cdf
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    # solve pdf of the distribution
    pdf = stats.burr.pdf(x, *burr_parameters)

    # solve cdf of the distribution
    cdf = stats.burr.cdf(x, *burr_parameters)
    #cdf_data = stats.burr.cdf(data, *burr_parameters) # solve cdf of corresponding data

    # Calculate tolerance interval
    burr_interval = stats.burr.interval(0.90, *burr_parameters)

    # Calculate p-value with KS test
    distribution_burr = stats.burr
    d_burr, pvalue_burr = stats.kstest(data, distribution_burr.cdf, burr_parameters)
    return burr_interval, pvalue_burr, burr_parameters, pdf, cdf

# Read .csv file
def read_csv(fname):
    data = []
    with open(fname, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data_tem = np.fromstring(row[0],dtype=float, sep=',')
            data = np.asarray(np.append(data, data_tem))
    return data


def fit_models(data):
    functions = [fit_normal, fit_lognormal, fit_weibull, fit_beta, fit_burr]
    stats_fncs = [stats.norm, stats.lognorm, stats.weibull_min, stats.beta, stats.burr]
    names = ['normal','lognormal', 'weibull', 'beta', 'burr']
    stats_fnc = [norm, lognorm, weibull_min, beta, burr]

    # Plot the histogram.
    y, bins, p = plt.hist(data, bins=20, density=True, alpha=0.6, color='g')

    # Define x axis for pdf plotting
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    # Model fitting with 5 different models
    for i in range(5):
        print('fitting data to '+names[i]+' distribution')
        interval, pvalue, params, pdf, cdf = functions[i](data)
        # Check based P-value
        if pvalue > 0.05:
            mean2 = stats_fnc[i].mean(*params)
            std2 = stats_fnc[i].std(*params)
            strs = ['Data fits to ' +names[i]+' distribution',
                    'Perform the Kolmogorov-Smirnov test for goodness of fit',
                    'P-value is ' + '%.4f' % pvalue,
                    'P-value > 0.05 => Do not reject the null hypothesis at a significance level 95%',
                    'Maximum Likelihood Estimator for distribution mean is ' + '%.4f' % mean2,
                    'Maximum Likelihood Estimator for distribution STD is ' + '%.4f' % std2,
                    '95% of data lies below '+ '%.4f' % interval[1]]
            for str in strs:
                print(str)

            # Plot distribution pdf
            plt.plot(x, pdf, linewidth=2, label=names[i])
            plt.legend()
            plt.title('Distribution PDF with data histogram')

            # Plot distribution cdf
            fig2 = plt.figure()
            plt.plot(x, cdf, color='black', linewidth=2, label=names[i]+' cdf')
            plt.plot([interval[1], interval[1]],[0,0.95],color='red', linestyle='dashed',label='95% upper limit')
            plt.legend()
            plt.title('Distribution CDF')
            break
    # Save results to folder "C:\\Statistical Analysis Results"
    #if not os.path.exists("C:\\Statistical Analysis Results\\"):
    #    os.makedirs("C:\\Statistical Analysis Results\\")
    #filename = open("C:\\Statistical Analysis Results\\Results.txt", "w")
    #for str in strs:
    #    filename.write(str + "\n")
    #filename.close()
    plt.show()
    return interval[1], names[i]


def fit_models_np_plot(data):
    functions = [fit_normal, fit_lognormal, fit_weibull, fit_beta, fit_burr]
    stats_fncs = [stats.norm, stats.lognorm, stats.weibull_min, stats.beta, stats.burr]
    #names = ['normal','lognormal', 'weibull', 'beta', 'burr']
    stats_fnc = [norm, lognorm, weibull_min, beta, burr]

    # Plot the histogram.
    #y, bins, p = plt.hist(data, bins=20, density=True, alpha=0.6, color='g')

    # Define x axis for pdf plotting
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    # Model fitting with 5 different models
    for i in range(5):
        interval, pvalue, params, pdf, cdf = functions[i](data)
        # Check based P-value
        if pvalue > 0.05:
            mean2 = stats_fnc[i].mean(*params)
            std2 = stats_fnc[i].std(*params)
            break

    return interval[1]


def fit_models_np_plot_mean_std(data):
    functions = [fit_normal, fit_lognormal, fit_weibull, fit_beta, fit_burr]
    stats_fncs = [stats.norm, stats.lognorm, stats.weibull_min, stats.beta, stats.burr]
    #names = ['normal','lognormal', 'weibull', 'beta', 'burr']
    stats_fnc = [norm, lognorm, weibull_min, beta, burr]

    # Plot the histogram.
    #y, bins, p = plt.hist(data, bins=20, density=True, alpha=0.6, color='g')

    # Define x axis for pdf plotting
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    # Model fitting with 5 different models
    for i in range(5):
        interval, pvalue, params, pdf, cdf = functions[i](data)
        # Check based P-value
        if pvalue > 0.05:
            mean2 = stats_fnc[i].mean(*params)
            std2 = stats_fnc[i].std(*params)
            break

    return interval[1], mean2, std2