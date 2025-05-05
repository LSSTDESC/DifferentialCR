Repository for developing and testing code to calculate differential chromatic refraction effects on survey data. 

This repository is based on a project completed by Matthew Lugatiman (contact details below). 
It has been uploaded to GitHub and further modified by Maya Redden (msredden@stanford.edu).

The "baseline_v3.3_10yrs.db" file used in airmass.ipynb (and other .db files) can be found at https://s3df.slac.stanford.edu/data/rubin/sim-data/ 
--> sims_featureScheduler_runs3.3/baseline/baseline_v3.3_10yrs.db

Original README by Matthew:
dcr_utils.py is all the code that I used to make each plot.

filter_files is a directory with all the filters that I manually call in my code (might be
Different when you use MAF)

airmass.ipnyb is a neat python notebook that Sid gave me to use the cursor to select visits from a specific database.

maf-egfootprint-example.html is a python tutorial to applying the cosmology cuts provided by Humna

quick_example contains an example of when I used the MAF to apply cosmology cuts and looked at the ellipticity quantiles at each filter. This uses a lot of functions in dcr_utils and I thought it would be helpful to incorporate.

Feel free to message me on slack @Matthew Lugatiman or email mluga002@ucr.edu in case you have any questions!

