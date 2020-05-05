These scripts to run the integrations were specific to our cluster etc., but we provide them for completeness. The functions to generate the orbital configurations (in runfunctions.py) should work generically.

runresonantscript.py generating and submitted jobs to the cluster for resonant configurations
runrandomscript.py did the same for the random configuratons
gen_whfastcomparison_commands.txt lists the commands used to make a comparison between WHFast and IAS15 with different whfast timesteps
whfastias15comparison.ipynb compares the integrations (probably have to fix paths)
