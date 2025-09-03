# Repository for developing and testing code to calculate differential chromatic refraction effects on survey data. 

This repository is based on a project completed by Matthew Lugatiman (contact details below). 
It has been uploaded to GitHub and further modified by Maya Redden (msredden@stanford.edu).

## Notebooks
- 'SEDAnalysis.ipynb' 
    - Propagates stellar and galactic SEDs through filter bands and then to distributions of refraction angle due to DCR (dN/dR).
    - Shows the distribution of DCR measurables (mean refraction angle E(R) and distribution dN/dR widths) for the set SEDs.
    - Attempts to show relationship between in-band SED slope and mean refraction angle within that band.
- 'DCRSecondStacker.ipynb'
    - Used for developing and testing a new MAF stacker class that will add a column(s) to OpSim .db files with estimated DCR second moment additive biases for an "average" galaxy.
    - Calculates the DCR second moment "magnitudes", $I_{DCR_{45}}$, that will be used in the new stacker.
    - Defines and tests the new stacker class with a real .db file. 
- 'airmass.ipynb'
    - Plots airmass distributions for a given .db file. 



## Data

- The "baseline_v3.3_10yrs.db" file used in airmass.ipynb (and other .db files) can be found at https://s3df.slac.stanford.edu/data/rubin/sim-data/ --> sims_featureScheduler_runs3.3/baseline/baseline_v3.3_10yrs.db

Simulated galaxy and stellar SEDs are used to evaluate DCR 
- 1000 simulated Galaxy SEDs w/ r magnitudes less than 26: '1000SEDs.txt'
    - first row is wavelengths in nanometers
    - next 1000 rows are the SED values in photons/nm/cm^2/s
- 4000 simulated Star SEDs w/ magnitudes between 17 and 22 from SkyCatalogs (paths preserved from their original location in NERSC)
    - Two different sets of SEDs, 'kurucz' and 'phoSimMLT', stored in 'starSEDs/global/cfs/cdirs/descssim/lsst/data/sim_sed_library/starSED/'
    - list of stellar SED paths in 'starSED_files_list.txt'


# Original README by Matthew:
dcr_utils.py is all the code that I used to make each plot.

filter_files is a directory with all the filters that I manually call in my code (might be
Different when you use MAF)

airmass.ipnyb is a neat python notebook that Sid gave me to use the cursor to select visits from a specific database.

maf-egfootprint-example.html is a python tutorial to applying the cosmology cuts provided by Humna

quick_example contains an example of when I used the MAF to apply cosmology cuts and looked at the ellipticity quantiles at each filter. This uses a lot of functions in dcr_utils and I thought it would be helpful to incorporate.

Feel free to message me on slack @Matthew Lugatiman or email mluga002@ucr.edu in case you have any questions!

