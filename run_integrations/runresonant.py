from runfunctions import run_resonant
import sys

seed = int(sys.argv[1])
runstring = sys.argv[2]
run_resonant(seed, runstr=runstring) # run system
run_resonant(seed, shadow=True, runstr=runstring) # run shadow system
