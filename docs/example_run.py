import os
cwd = os.getcwd()
from sherpa.astro import ui
import numpy as np
from sherpa.astro import xspec

#run the code below within the data folder

ui.load_data("/6122.pi") #load the observation data from the folder
ui.ignore() #ignore all data first...
ui.notice(0.4,7.0) #...and then explicitly read data to be included in the fitting.
#This selects data in the range of 0.4-7.0 keV, which is where Chandra is most sensitive.

#load the files 6122.corr.arf and 6122.rmf from the data folder
ui.get_arf() #returns the ARF associated with the PHA dataset
ui.get_rmf() #returns the RMF associated with the PHA dataset

#put the data in log space
ui.set_xlog()
ui.set_ylog()

#set the model for the fit. In this case, we use Model 2v from Acharya et al. 2023
ui.set_source((xspec.XSvapec.t1 + xspec.XSvapec.t2) * xspec.XSTBabs.mdl)

#set the statistic to the cash statistic
ui.set_stat("cstat")

#fix the value of the neutral H column density of XSTBabs.mdl to 1e19 cm-2.
mdl.nH = 0.001
freeze(mdl.nH)

#now, we tie the abundances of the 2 temperature components to vary together
t2.C = t1.C ; t1.N = t1.N ; t2.O = t1.O ; t2.Ne = t1.Ne ; t2.Mg = t1.Mg ; t2.Al = t1.Al ; t2.Si = t1.Si ; 
t2.S = t1.S ; t2.Ar = t1.Ar ; t2.Ca = t1.Ca ; t2.Fe = t1.Fe ; t2.Ni = t1.Ni ; t2.He = t1.He

#now, we tie some of the elements together according to Model 2v
t1.C = t1.S = t1.O
t1.Ar = t1.N
t1.Mg = t1.Al = t1.Si = t1.Ca = t1.Ni = t1.Fe

#now, O, N, Ne and Fe will be allowed to vary while performing the fitting
ui.thaw(t1.N,t1.O,t1.Ne,t1.Fe)

#run the fit() function a few times
for i in range(5):
    ui.fit()

#Now we use the full code csresid.py to generate plots from the bottom row of Figure 12 of Acharya et al. 2023
import csresid as csr

pct_cusum,pval = csr.CuSumRes(outdir=cwd,outroot='',domcmc=False,n_burn=3000,get_draws_niter=7000,nsim=300, 
Emin=0.4,Emax=7.0,figsize=(10,6),lw=2.5,modellabel="2-T Variable Abundance",boundscolor='black',
modelcolor='tab:red',histcolor='tab:green',bins =30,figformat='pdf')

print"The pct_cusum is ", round(pct_cusum,2), "%")
print("The p value is ",round(pval,2)) 
#if everything is done correctly, this would be 13.5% and 0.09 respectively



