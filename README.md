# csresid
## Cumulative Sum of Residuals analysis of X-ray Spectra
The primary purpose of this code is to evaluate the residuals of the best-fit model versus the observational data, and verify if the behaviour is acceptable. For doing this, we first generate a pool of simulated residuals. This can be done using 2 methods.
# Method 1: 
Using fake_pha, we can generate a mock dataset, run fit() on it to obtain residuals versus this simulated dataset, and repeat the process until we have as many sets of residuals as we need.
# Method 2: 
Sample best-fits from MCMC iterations obtained by running pyBloCXS called via get_draws. The residuals between these best-fits and observational data is then used. Once we have a set of simulated residuals, we plot their cumulative sum versus energy, and find their 5-95th percentile range. Then we compare the cumulative sum of residuals of our obtained best-fit on the real data with this. Ideally, ~10% points of this curve would lie beyond the 5-95th percentile range. We calculate this number and report it along with the plot. Secondly, we note the distance by which they go beyond the 5-95th percentile range. We call the sum of these lengths as the excess area. We calculate this for all the simulated residuals to obtain a histogram. We also calculate this for the best-fit on the observational data.
#The p-value of excess area should be >0.05 as the best fit to the real data should not be any different from the best fit to simulated data.

## In this code, the function CuSumRes() is the master control, which calls all other functions.
In gensim(), you can choose whether you want to use MCMC draws for generating simulations using get_draws(), or if you want to just use fake_pha() to generate simulations.
If you choose to run MCMC, mcmcsims() will call diagnosticplot_maker(), that plots the behaviour of parameters and stats versus iterations.
If you don't choose MCMC, you would be using fakephasims().
plot_res() is used to plot the trend of cumulative sum of residuals versus energy, and
excessarea() plots the excess beyond the 5th and 95th percentile.

An example call for the full code would be as the following:
## Method 1: Using fake_pha() to generate simulated data.
pct_CuSum,pval_area = CuSumRes(outdir='/path/to/where/you/want/to/save/your/file',outroot='test1_',domcmc=False,Emin=0.4,Emax=7.0,nsim=300, figsize=(10,6),lw=2.5,modellabel='Model',boundscolor='black',modelcolor='tab:red',histcolor='tab:green',bins =30,figformat='pdf')
## Method 2: Using get_draws() to generate best-fit parameters.
 pct_CuSum,pval_area = CuSumRes(outdir='/path/to/where/you/want/to/save/your/file',outroot='test2_',domcmc=True,Emin=0.4,Emax=7.0,nsim=300,n_burn=3000,get_draws_niter=7000, figsize=(10,6),lw=2.5,modellabel='Model',boundscolor='black',modelcolor='tab:red',histcolor='tab:green',bins =30,figformat='pdf')
#
call help() to get details of the different variables
#

## Output files generated:
## Method 1:
   (i) model_on_realdata.txt: After applying the calculated best-fit to the data, this saves a text file with the following arrays:
       x values of data (in keV), y values of data (in counts/sec/keV), y values of model (in counts/sec/keV), y values of residuals, cumulative sum of residuals.
   (ii) csum_i.txt: For each simulated dataset generated using fake_pha(), a best-fit is calculated, and the following arrays are saved to a text file:
       lower bound of x values of the fit (in keV), upper bound of x values of the fit (in keV), y values for the simulated data (in counts/sec/keV), y value for
       the model fit to the simulated data (in counts/sec/keV), y values of residuals, cumulative sum of residuals. Both lower and upper boounds of x values of the
       fit are provided if needed for additional error calculation.
   (iii) residuals_for_CuSum: a figure with the cumulative sum of the best-fit on real data shown in the color chosen using modelcolor (default is 'tab:red'),
       and the 5-95th percentile bounds of the cumulative sum of the best-fits on the simulated data shown in the color chosen using boundscolor (default is 'black').
   (iv) excessarea_for_CuSum: a figure with the sum of deviations of the real data best-fit cumulative sum beyond the 5-95th percentile bounds (from the above figure)
       shown as a vertical line in the color chosen using modelcolor (default is 'tab:red'), and a histogram for the same for best-fit cumulative sum for simulated
       data shown in the color chosen using histcolor (default is 'tab:green') with number of bins chosen using the variable bins (default is 30).
## Method 2:
   (i): model_on_realdata.txt: same as Method 1, except that we check the Cash-statistic for each iteration of MCMC, and replace the best-fit parameters with an
       iteration which gives a lower Cash-statistic before saving this file.
   (ii): csum_i.txt: same as Method 1, except the MCMC iteration of the fit is applied to the real data for calculating residuals for each iteration.
   (iii): get_drawsoutput.fits: saves all MCMC iterations with columns of all unfrozen parameters, stats and accept, as generated by get_draws().
   (iv): diagnostic_plots: a plot of all parameters and stats versus number of iterations given by the get_draws_niter variable, for diagnostics of the MCMC runs.
   (v): residuals_for_CuSum: same as Method 1.
   (vi): excessarea_for_CuSum: same as Method 1.


Anshuman Acharya (January 2022)

v1.0
#
Reference: see Acharya et al. (2023), http://arxiv.org/abs/2211.01011
### For any questions regarding usage of any functions, feel free to write to the authors of the code.

