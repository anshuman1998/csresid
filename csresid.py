####Cumulative Sum of Residuals analysis of X-ray Spectra####
#The primary purpose of this code is to evaluate the residuals of the best-fit model
#versus the observational data, and verify if the behaviour is acceptable.
#For doing this, we first generate a pool of simulated residuals. This can be done using 2 methods.
#Method 1: Using fake_pha, we can generate a mock dataset, run fit() on it to obtain residuals versus this simulated dataset,
#and repeat the process until we have as many sets of residuals as we need.
#Method 2: Sample best-fits from MCMC iterations obtained by running pyBloCXS called via get_draws.
#The residuals between these best-fits and observational data is then used.
#
#Once we have a set of simulated residuals, we plot their cumulative sum versus energy, and find their 5-95th percentile range.
#Then we compare the cumulative sum of residuals of our obtained best-fit on the real data with this.
#Ideally, ~10% points of this curve would lie beyond the 5-95th percentile range. We calculate this number and report it along with the plot.
#Secondly, we note the distance by which they go beyond the 5-95th percentile range. We call the sum of these lengths as the
#excess area. We calculate this for all the simulated residuals to obtain a histogram. We also calculate this for the best-fit on the observational data.
#The p-value of excess area should be >0.05 as the best fit to the real data should not be any different from the best fit to simulated data.

##In this code, the function CuSumRes() is the master control, which calls all other functions
##in gensim(), you can choose whether you want to use MCMC draws for generating simulations
##using get_draws(), or if you want to just use fake_pha() to generate simulations.
##If you choose to run MCMC, mcmcsims() will call diagnosticplot_maker(), that plots the behaviour of parameters and stats versus iterations.
#If you don't choose MCMC, you would be using fakephasims().
##plot_res() is used to plot the trend of cumulative sum of residuals versus energy, and
##excessarea() plots the excess beyond the 5th and 95th percentile.
##########################
#
#An example call for the full code would be as the following:
#Method 1: Using fake_pha() to generate simulated data.
#pct_CuSum,pval_area = CuSumRes(outdir='/path/to/where/you/want/to/save/your/file',outroot='test1_',domcmc=False,Emin=0.4,Emax=7.0,nsim=300, figsize=(10,6),lw=2.5,modellabel='Model',boundscolor='black',modelcolor='tab:red',histcolor='tab:green',bins =30,figformat='pdf')
#Method 2: Using get_draws() to generate best-fit parameters.
# pct_CuSum,pval_area = CuSumRes(outdir='/path/to/where/you/want/to/save/your/file',outroot='test2_',domcmc=True,Emin=0.4,Emax=7.0,nsim=300,n_burn=3000,get_draws_niter=7000, figsize=(10,6),lw=2.5,modellabel='Model',boundscolor='black',modelcolor='tab:red',histcolor='tab:green',bins =30,figformat='pdf')
#########################
# call help() to get details of the different variables
#########################
#########################
#########################
#Output files generated:
#Method 1:
#   (i) model_on_realdata.txt: After applying the calculated best-fit to the data, this saves a text file with the following arrays:
#       x values of data (in keV), y values of data (in counts/sec/keV), y values of model (in counts/sec/keV), y values of residuals, cumulative sum of residuals.
#   (ii) csum_i.txt: For each simulated dataset generated using fake_pha(), a best-fit is calculated, and the following arrays are saved to a text file:
#       lower bound of x values of the fit (in keV), upper bound of x values of the fit (in keV), y values for the simulated data (in counts/sec/keV), y value for
#       the model fit to the simulated data (in counts/sec/keV), y values of residuals, cumulative sum of residuals. Both lower and upper boounds of x values of the
#       fit are provided if needed for additional error calculation.
#   (iii) residuals_for_CuSum: a figure with the cumulative sum of the best-fit on real data shown in the color chosen using modelcolor (default is 'tab:red'),
#       and the 5-95th percentile bounds of the cumulative sum of the best-fits on the simulated data shown in the color chosen using boundscolor (default is 'black').
#   (iv) excessarea_for_CuSum: a figure with the sum of deviations of the real data best-fit cumulative sum beyond the 5-95th percentile bounds (from the above figure)
#       shown as a vertical line in the color chosen using modelcolor (default is 'tab:red'), and a histogram for the same for best-fit cumulative sum for simulated
#       data shown in the color chosen using histcolor (default is 'tab:green') with number of bins chosen using the variable bins (default is 30).
#Method 2:
#   (i): model_on_realdata.txt: same as Method 1, except that we check the Cash-statistic for each iteration of MCMC, and replace the best-fit parameters with an
#       iteration which gives a lower Cash-statistic before saving this file.
#   (ii): csum_i.txt: same as Method 1, except the MCMC iteration of the fit is applied to the real data for calculating residuals for each iteration.
#   (iii): get_drawsoutput.fits: saves all MCMC iterations with columns of all unfrozen parameters, stats and accept, as generated by get_draws().
#   (iv): diagnostic_plots: a plot of all parameters and stats versus number of iterations given by the get_draws_niter variable, for diagnostics of the MCMC runs.
#   (v): residuals_for_CuSum: same as Method 1.
#   (vi): excessarea_for_CuSum: same as Method 1.
#########################

#Anshuman Acharya (January 2022)
#v0.920230704
#Reference: see Acharya et al. (2023), http://arxiv.org/abs/2211.01011

import os
cwd = os.getcwd()
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from sherpa.astro import ui
from astropy.table import Table
import time
from time import localtime, strftime

def gensim(outdir=cwd,outroot='',domcmc=False, n_burn=3000,get_draws_niter=7000,nsim=300,figformat='pdf'):
    if domcmc is True:
        print("using MCMC")
        mcmcsims(outdir,outroot,n_burn,get_draws_niter,nsim,figformat)
    else:
        print("not using MCMC")
        fakephasims(outdir,outroot,nsim)
    return True
        
def fakephasims(outdir=cwd,outroot="",nsim=300):
    nsim = int(nsim)
    print("to be run for ",nsim," simulations")
    modelname = ui.get_source() #get the fitting model
    ovals = [(p, p.val) for p in modelname.pars if not p.frozen] #save the names and values of parameters that are not frozen
    ovals_name = [p.fullname for p in modelname.pars if not p.frozen]
    ovals_numeric = [p.val for p in modelname.pars if not p.frozen] #save the numeric values of the parameters that are not frozen
    print("These are the results from the standard fit:")
    print(ovals_numeric)
    
    #ARF, RMF, Exposure time and energy range of data
    arf_file = ui.get_arf()
    rmf_file = ui.get_rmf()
    obs_time = ui.get_data().exposure
    elow = min(ui.get_data_plot().x)
    ehigh = max(ui.get_data_plot().x)
    
    ui.plot_fit_resid()
    #save the best fit model results
    datax,datay,modely,residualy = ui.get_data_plot().x, ui.get_data_plot().y,ui.get_model_plot().y,ui.get_resid_plot().y
    ui.save_arrays(str(outdir)+"/"+str(outroot)+"model_on_realdata.txt",[datax,datay,modely,residualy,np.cumsum(residualy)],clobber=True)
    #the arrays saved are: xvals of data, yvals of data, yvals of model,
    # yvals of residuals, cumulative sum of residuals
    
    #Run simulations of the data
    for iter in range(nsim):
        #Create the simulated dataset
        ui.fake_pha(1,arf_file,rmf_file,obs_time)
        ui.notice(elow,ehigh)
        #Now, we use the parameters of the fit with this simulated data to get the residuals
        ui.fit()
        ui.plot_fit_resid()
        ui.save_arrays(str(outdir)+"/"+str(outroot)+"csum_"+str(iter)+".txt",[ui.get_model_plot().xlo,ui.get_model_plot().xhi,
        ui.get_data_plot().y,ui.get_model_plot().y,ui.get_resid_plot().y,np.cumsum(ui.get_resid_plot().y)],clobber=True)
        #The arrays saved are:
        #xlo for the model fit to the simulated data
        #xhi for the model fit to the simulated data
        #y for the simulated data
        #y for the model fit to the simulated data
        #residual, i.e., the difference of the above 2 y values
        #cumulative sum of residuals
        ################################
        ct = 0
        for p in modelname.pars:
            if not p.frozen:
                p.val = ovals_numeric[ct]
                ct+=1
        #Reset to original model so that when quitting out of this script, best fit parameters are retained
        ###Anshuman Acharya thanks Doug Burke for the code to loop through model parameters and replacing them as needed.
    return True
        
def mcmcsims(outdir=cwd,outroot='',n_burn=3000,get_draws_niter=7000,nsim=300,figformat='pdf'):
    start_time = time.clock()
    
    #First ensure all numbers are converted to integers
    n_burn,get_draws_niter,nsim = int(n_burn),int(get_draws_niter),int(nsim)
    
    original_cstat = ui.get_stat_info()[0].statval #save the cstat for the standard fit output
    
    ui.covar() #check covariance
    modelname = ui.get_source() #get the fitting model
    ovals = [(p, p.val) for p in modelname.pars if not p.frozen] #save the names and values of parameters that are not frozen
    ovals_name = [p.fullname for p in modelname.pars if not p.frozen]
    ovals_numeric = [p.val for p in modelname.pars if not p.frozen] #save the numeric values of the parameters that are not frozen

    print("These are the results from the standard fit:")
    print(ovals_numeric)

    #Run pyBLoCXS
    stats,accept,params=ui.get_draws(1,niter=get_draws_niter+n_burn)
    
    #Save the pyBLoCXS output
    headerlist = []
    for pr in range(len(params)):
        headerlist.append(ovals_name[pr])
    headerlist.append("stats")
    headerlist.append("accept")
    alldata = params.tolist()
    alldata.append(stats)
    alldata.append(accept)
    get_drawstable=Table(alldata,names=headerlist)
    
    get_drawstable.write(str(outdir)+"/"+str(outroot)+"get_drawsoutput.fits", format='fits',overwrite=True)
    del alldata
      
    time_taken_formcmc = time.clock() - start_time
    print("making diagnostic plots next:")
    if diagnosticplot_maker(ovals_numeric,outdir,outroot,figformat,time_taken_formcmc):
        print("Done!")
    
    #Now we go through the pyBLoCXS output after crossing n_burn iterations, and use nsim values as fit parameters
    for iter in range(nsim):
        param_notfrozen = [] #save the value of the not frozen parameter in each iteration taken
        for pr in range(len(params)):
            #if get_draws crosses the min/max value of a parameter, replace it with that min/max limit
            if params[pr][int(n_burn+get_draws_niter-iter)] >= ovals[pr][0].min and params[pr][int(n_burn+get_draws_niter-iter)] <= ovals[pr][0].max:
                param_notfrozen.append(params[pr][int(n_burn+get_draws_niter-iter)])
            elif params[pr][int(n_burn+get_draws_niter-iter)] < ovals[pr][0].min:
                param_notfrozen.append(ovals[pr][0].min)
            else:
                param_notfrozen.append(ovals[pr][0].max)
        #to calculate cstat, replace the parameter values from the pyBLoCXS iteration into the model
        ct = 0
        print("The fit parameters for this iteration are:")
        for p in modelname.pars:
            if not p.frozen:
                p.val = param_notfrozen[ct]
                ct+=1
                print(p.val)
        print("----------------------------")
        ui.plot_fit_resid() #find residuals for the fit parameters of the (n_burn+get_draws_niter-iter)th iteration of get_draws versus the data
        #save the cusum
        ui.save_arrays(str(outdir)+"/"+str(outroot)+"csum_"+str(iter)+".txt",[ui.get_model_plot().xlo,ui.get_model_plot().xhi,
        ui.get_data_plot().y,ui.get_model_plot().y,ui.get_resid_plot().y,np.cumsum(ui.get_resid_plot().y)],clobber=True)
        
        #if the cstat with the pyBLoCXS iteration values is higher than the standard fit cstat, restore values in the model
        if ui.get_stat_info()[0].statval > original_cstat:
            ct = 0
            print("Cstat is: ", ui.get_stat_info()[0].statval)
            for p in modelname.pars:
                if not p.frozen:
                    p.val = ovals_numeric[ct]
                    print("Reset to ",ovals_numeric[ct])
                    ct+=1
        #but if it is lower, then this pyBLoCXS iteration becomes the new best fit
        else:
            original_cstat = ui.get_stat_info()[0].statval
            ovals_numeric = []
            for p in modelname.pars:
                if not p.frozen:
                    ovals_numeric.append(p.val)
                
    print("New fit parameters are:")
    for p in modelname.pars:
        if not p.frozen:
            print(p.val)
    ###Anshuman Acharya thanks Doug Burke for the code to loop through model parameters and replacing them as needed.
    print("With a cstat of:",ui.get_stat_info()[0].statval)
                    
    ui.plot_fit_resid()
    #save the best fit model results
    datax,datay,modely,residualy = ui.get_data_plot().x, ui.get_data_plot().y,ui.get_model_plot().y,ui.get_resid_plot().y
    ui.save_arrays(str(outdir)+"/"+str(outroot)+"model_on_realdata.txt",[datax,datay,modely,residualy,np.cumsum(residualy)],clobber=True)
    #the arrays saved are: xvals of data, yvals of data, yvals of model,
    # yvals of residuals, cumulative sum of residuals
    print("Done with generating sims")
    return True #n_burn+get_draws_niter, nsim

######The diagnostic plots for the MCMC runs of get_draws() in mcmcsims#######
def diagnosticplot_maker(ovals_numeric,outdir=cwd,outroot='',figformat='pdf',time_taken_formcmc=0.0):
    gdop = Table.read(str(outdir)+"/"+str(outroot)+"get_drawsoutput.fits")
    
    diagnostic_x = np.arange(1,len(gdop)+0.1,1)
    numplots = len(gdop[0])
    fig = plt.figure(figsize=(30,20))
    gs = gridspec.GridSpec(ncols=2, nrows=int(numplots/2), figure=fig,width_ratios=[1,1],height_ratios=np.ones(int(numplots/2)))
    ct,acceptancerate = 0, 0
    for g,colname in zip(gs,gdop.colnames):
        if ct<(numplots-2):
            ax = plt.subplot(g)
            ax.tick_params(right=True,top=True,which='both',direction='in')
            ax.plot(diagnostic_x,gdop[colname],lw=3,alpha=0.5,color='tab:blue')
            ax.axhline(y=np.mean(gdop[colname]),lw=3,label='avg',color='k',alpha=0.8,ls='--')
            ax.axhline(y=ovals_numeric[ct],lw=3,label='standard fit',color='k',alpha=0.8)
            ax.set_xlabel("niter",fontsize=16)
            ax.set_ylabel(colname,fontsize=16)
            ax.set_xlim(-100,max(diagnostic_x)+100)
            ax.legend(loc="upper left",framealpha=0.9,fontsize=14,ncol=2)
        elif ct==numplots-2:
            ax = plt.subplot(g)
            ax.tick_params(right=True,top=True,which='both',direction='in')
            ax.plot(diagnostic_x,gdop[colname],lw=3,alpha=0.5,color='tab:orange')
            ax.axhline(y=np.mean(gdop[colname]),lw=3,label='avg',color='k',alpha=0.8,ls='--')
            ax.set_xlabel("niter",fontsize=16)
            ax.set_ylabel(colname,fontsize=16)
            ax.set_xlim(-100,max(diagnostic_x)+100)
            ax.legend(loc="upper left",framealpha=0.9,fontsize=14)
        else:
            ax = plt.subplot(g)
            ax.axis('off')
            for i in gdop[colname]:
                if i:
                    acceptancerate+=1
            acceptancerate = round(acceptancerate/len(gdop[colname]),6)
            ax.text(0.05,0.6,"CuSumRes: \na code for analysing \nthe cumulative sum of residuals",fontsize=18,backgroundcolor='#BBBBBB',family='cursive')
            ax.text(0.05,0.4,"MCMC draws took "+str(round(time_taken_formcmc,3))+" seconds",fontsize=16,backgroundcolor='#BBBBBB')
            ax.text(0.05,0.3,"with an accept = "+str(acceptancerate),fontsize=16,backgroundcolor='#BBBBBB')
            ax.text(0.05,0.2,"generated at: "+strftime("%Y-%m-%d %H:%M:%S", localtime()),fontsize=16,backgroundcolor='#BBBBBB')
        ct+=1
    plt.savefig(str(outdir)+"/"+str(outroot)+"diagnostic_plots."+figformat, dpi=300, format=figformat,bbox_inches="tight")
    del diagnostic_x,ct,numplots,gs,fig
    return True

#Now, to plot the residuals from model to actual data, vs model to simulated datasets
#Also returns the percentage of bins beyond the 5-95th percentile range
def plot_res(outdir=cwd,outroot='',nsim=300,Emin=0.4,Emax=7.0,figsize=(10,6),lw=2.5,modellabel="Model",boundscolor='black',modelcolor='tab:red',figformat='pdf'):
    
	#Define the plot size
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    nsim = int(nsim)
    print(nsim)
    #Get the cumulative sums of all the simulated datasets
    csum = []
    print("Started plotting routine for CuSum")
    for i in range(nsim):
        file = np.genfromtxt(str(outdir)+"/"+str(outroot)+"csum_"+str(i)+".txt")
        csum.append(file[:,5])
    
    csum = np.array(csum).T
    #Calculate the 5th and 95th percentile bounds
    bound_5,bound_95 = [], []
    for r in csum:
        bound_5.append(np.percentile(r,5))
        bound_95.append(np.percentile(r,95))
        
    #Get the energy bins and the cumulative sum of model applied to real data
    file = np.genfromtxt(str(outdir)+"/"+str(outroot)+"model_on_realdata.txt")
    xvals = file[:,0]
    model_yvals = file[:,4]
        
    #Plot the 5-95th percentile bounds
    ax.plot(xvals,bound_5,color=boundscolor,linestyle='--',lw=lw,label="5-95th Percentile")
    ax.plot(xvals,bound_95,color=boundscolor,linestyle='--',lw=lw)
    #Plot the model cusum
    ax.plot(xvals,model_yvals,color=modelcolor,lw=2*lw,label=modellabel)
    ax.axhline(y=0.0,color='black')
        
    ax.set_ylabel("CuSum [ct s$^{-1}$ keV$^{-1}$]",fontsize=25)
    ax.set_xlabel("Energy [keV]",fontsize=25)
    ax.set_xlim(left=Emin,right=Emax)
    ax.legend(loc='upper right',fontsize=25)

    #Find the number of bins beyond the bounds in the bin range of interest.
    #The range of interest should include the energy range within which 99% of source counts exist.
    ct=0
    for i,j,k,xaxis in zip(model_yvals,bound_5,bound_95,xvals):
        if Emin<xaxis<=Emax:
            if i>0 and i>=k:
                ct+=1
            if i<0 and i<=j:
                ct+=1
    pct_cusum = 100 *ct/len(bound_5)
    
    plt.savefig(str(outdir)+"/"+str(outroot)+"residuals_for_CuSum."+figformat, dpi=300, format=figformat,bbox_inches="tight")
    
    return pct_cusum
    
#Find the p-value of excess area beyond the 5-95th percentile bounds.
#plots the value against the histogram of applying the model to simulated datasets
#and also returns the pvalue.
def excessarea(outdir=cwd,outroot='',nsim=300,figsize=(10,6),bins=30,lw=2.5,modellabel="Model",histcolor='tab:green',modelcolor='tab:red',figformat='pdf'):
    print("Plotting the excess area")
    #Define the plot size
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    nsim = int(nsim)
    print(nsim)
    #Get the cumulative sums of all the simulated datasets
    csum = []
    for i in range(nsim):
        file = np.genfromtxt(str(outdir)+"/"+str(outroot)+"csum_"+str(i)+".txt")
        csum.append(file[:,5])
    csum_arr = csum #Keep a copy of the original as it is easier to use
    csum = np.array(csum).T
    #Calculate the 5th and 95th percentile bounds
    bound_5,bound_95 = [],[]
    for r in csum:
        bound_5.append(np.percentile(r,5))
        bound_95.append(np.percentile(r,95))
        
    #Get the energy bins and the cumulative sum of model applied to real data
    file = np.genfromtxt(str(outdir)+"/"+str(outroot)+"model_on_realdata.txt")
    xvals = file[:,0]
    model_yvals = file[:,4]
    
    #Find the extent by which cumulative sum at a particular energy bin for the model applied to real data
    #is beyond the 95th percentile. If it is not, append 0.
    beyond95 = []
    for c,b95 in zip(model_yvals,bound_95):
        if c-b95>0:
            beyond95.append(c-b95)
        else:
            beyond95.append(0.0)
            
    #Find the extent by which cumulative sum at a particular energy bin for the model applied to real data
    #is beyond the 5th percentile. If it is not, append 0.
    beyond05 = []
    for c,b05 in zip(model_yvals,bound_5):
        if c-b05<0:
            beyond05.append(c-b05)
        else:
            beyond05.append(0.0)
        
    #Sum the above two arrays
    sum_data = sum(beyond95)+abs(sum(beyond05))
        
    #Repeat the above process for all the simulated datasets
    ct = 0
    sim_beyond_vals = []
    for arr in csum_arr:
        ct+=1
        sim_beyond = 0
        for c,b95,b05 in zip(arr,bound_95,bound_5):
            if (c-b95)>0:
                sim_beyond+=(c-b95)
            if (c-b05)<0:
                sim_beyond+=(b05-c)
            
        sim_beyond_vals.append(sim_beyond)
        if ct%100==0:
            print("Done for ",ct, "simulation")
        
    #Plot the histogram of excess for the model applied to simulated data
    hist_sim = ax.hist(sim_beyond_vals,bins=bins,color=histcolor,lw=lw,histtype='step',zorder=-32,label="Simulated Data")
    #Plot the 95th percentile
    ax.axvline(x=np.percentile(sim_beyond_vals,95),lw=lw,linestyle="--",color="k",alpha=0.8,zorder=-16,label="95th Percentile")
    #Plot the value of excess for the model applied to real data
    ax.axvline(x=sum_data,lw=lw,color=modelcolor,label=r"Observed p$_{\rm area}$")
    
    ax.set_xlabel("Excess area [ct s$^{-1}$]",fontsize=25)
    ax.set_ylabel("Frequency",fontsize=25)
    ax.set_yscale("log")
    ax.legend(loc='upper right',fontsize=25)
    
    #Calculate the p-value of the excess area of applying the mode to real data
    #as compared to applying it to simulated data
    sim_beyond_vals.append(sum_data)
    idx   = np.argsort(sim_beyond_vals)
    sim_beyond_vals=np.array(sim_beyond_vals)[idx]
    sim_beyond_vals = sim_beyond_vals.tolist()
    
    pval = (1-(sim_beyond_vals.index(sum_data)/nsim))
    #print((1-(sim_beyond_vals.index(sum_data)/nsim))," is the p-value for this fit")
    
    plt.savefig(str(outdir)+"/"+str(outroot)+"excessarea_for_CuSum."+figformat, dpi=300, format=figformat,bbox_inches="tight")
    
    return pval

def CuSumRes(outdir=cwd,outroot='',domcmc=False,n_burn=3000,get_draws_niter=7000,nsim=300, Emin=0.4,Emax=7.0,figsize=(10,6),lw=2.5,modellabel="Model",boundscolor='black',modelcolor='tab:red',histcolor='tab:green',bins =30,figformat='pdf'):

    gensim(outdir,outroot,domcmc,n_burn,get_draws_niter,nsim,figformat)

    pct_cusum = plot_res(outdir,outroot,nsim,Emin,Emax,figsize,lw,modellabel,boundscolor,modelcolor,figformat)
    pval = excessarea(outdir,outroot,nsim,figsize,bins,lw,modellabel,histcolor,modelcolor,figformat)
    
    print("The % CuSum value is (ideally ~10%):",pct_cusum)
    print("And the p-value of excess area is (ideally >0.05):",pval)
    
    return pct_cusum,pval
    
def help():
    print("The following functions are available in this package:")
    print("1. CuSumRes(): This is the master control function that calls all other functions. It can be used as the following")
    print("CuSumRes(outdir='/path/to/your/file',outroot='',domcmc=False,n_burn=3000,Emin=0.4,Emax=7.0,get_draws_niter=7000,nsim=300,")
    print("figsize=(10,6),lw=2.5,modellabel='Model',boundscolor='black',modelcolor='tab:red',histcolor='tab:green',bins =30,figformat='pdf')")
    print("Here the values listed are also set as defaults, so the secondary options for the figure can be skipped to generate plots as done in Acharya et al. (2023)")
    print("Output of this function is percentage of cumulative sum points beyond the 5-95th percentile range (ideally ~10%) and p-value (ideally >0.05).")
    print("inputs that can be provided to CuSumRes are the following:")
    print("     (i): outdir: The output directory where the outputs should be saved. The default is the folder from where this code has been called.")
    print("     (ii): outroot: The starting prefix for all output files generated. The default is '', so no prefix will be added unless specified.")
    print("     (iii): domcmc: If False, simulated datasets are generated using fake_pha. If True, get_draws is used, and additional parameters for number of iterations and number of iterations needed for burning are to be defined. The default is False.")
    print("     (iv): n_burn: The number of iterations needed for burning while running MCMC. Default is 3000.")
    print("     (v): get_draws_niter: The number of iterations for which the MCMC is run after burning. Default is 7000.")
    print("     (vi): nsim: The number of simulated datasets you would like to generate using fake_pha or the MCMC runs. Default is 300.")
    print("     (vii): Emin, Emax: The minimum and maximum energy range over which the cumulative sum is calculated. The default is 0.4 and 7.0, in keV.")
    print("     (viii): figsize: This takes the dimensions of the plots for the cumulative sum and the excess area as a tuple. Default is (10, 6).")
    print("     (ix): lw: The linewidth to be used in the plots. Default is 2.5.")
    print("     (x): modellabel= The label for the plot for the fitting model being tested. Default is 'model'.")
    print("     (xi): boundscolor, modelcolor: Color for the 5-95th percentile bounds in the cumulative sum plot (black by default), and the color for the model for the cumulative sum as well as excess area plots (red by default), respectively.")
    print("     (xii): histcolor, bins: Color of the histogram in the excess area plot (green by default) and the number of bins for the histogram.")
    print("     (xiii): figformat: Format in which the generated plots are to be saved. 'pdf' by default.")
    print("---------------------------------------------------------------------------------------------------")
    print("2. gensim(outdir='/path/to/your/file',outroot='',n_burn=3000,get_draws_niter=7000,nsim=300,domcmc=False,figformat='pdf')")
    print("choosing domcmc=False is the default, in which case the simulated datasets are generated using fake_pha. \nIf domcmc=True, then get_draws is used to generate simulated datasets, and a diagnostics plot is generated and saved in the format provided by figformat, which is 'pdf' by default.")
    print("---------------------------------------------------------------------------------------------------")
    print("3. fakephasims(outdir='/path/to/your/file',outroot='',nsim=300)")
    print("4. mcmcsims(outdir='/path/to/your/file',outroot='',n_burn=3000,get_draws_niter=7000,nsim=300,figformat='pdf') \ncalls diagnosticplot_maker that plots the behaviour of various parameters and stats versus iterations after burning.")
    print("---------------------------------------------------------------------------------------------------")
    print("5. plot_res(outdir='/path/to/your/file',outroot='',nsim=300,Emin=0.4,Emax=7.0,figsize=(10,6),lw=2.5,modellabel='Model',boundscolor='black',modelcolor='tab:red',figformat='pdf')")
    print("The output of this function is the plot of the cumulative sum of residuals versus energy for the best-fit versus the 5-95th percentile range for simulated datasets. It also returns the percentage of points beyond the 5-95th percentile range.")
    print("---------------------------------------------------------------------------------------------------")
    print("6. excessarea(outdir='/path/to/your/file',outroot='',nsim=300,figsize=(10,6),lw=2.5,modellabel='Model',histcolor='tab:green',bins=30,modelcolor='tab:red',figformat='pdf')")
    print("The output of this function is a plot of total area beyond the 5-95th percentile range for the best-fit on the observational data as well as a vertical line, with that for simulated data as a histogram. It also returns the p-value of the excess area for the best-fit on observational data as compared to simulated data.")
    print("---------------------------------------------------------------------------------------------------")
    print("7. diagnosticplot_maker(ovals_numeric,outdir='/path/to/your/file',outroot='',figformat='pdf',time_taken_formcmc=0.0)")
    print("ovals_numeric is the list of best-fit parameters for which the Cash-statistic is minimised, and time_taken_formcmc is just the time take to run the MCMC code. Both these values are supplied by the mcmcsims function.")
    print("---------------------------------------------------------------------------------------------------")
    print("For any confusion regarding usage of any functions, feel free to write to the authors of the code.")
    

