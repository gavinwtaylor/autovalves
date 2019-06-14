# autovalves

For an RL approach to autonomously control valves in chemical plants.

Gavin Taylor, US Naval Academy

Tom Adams, McMaster University


#To Train:
To train, run $WORKDIR/autovalves/learner/cstr_ex.py using the following command
mpiexec -np <numprocs> python cstr_ex.py <learning rate> <timestep> <entropy> <value coefficent> 

#To Evaluate:

#To see results:
In order to see the results from the evaluation, execute the statFinder.sh script in the autovalves directory.  This will execute python autovalves/simulator/expSun31/extractData.py and will print out the stats from the evalution in the stdout file for statFinder.sh in autovalves.
