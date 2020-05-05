from runfunctions import run_random
import sys

sim_id = int(sys.argv[1])
runstring = sys.argv[2]
run_random(sim_id, runstr=runstring) # run system
run_random(sim_id, shadow=True, runstr=runstring) # run shadow system
