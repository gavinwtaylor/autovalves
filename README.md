# autovalves

For an RL approach to autonomously control valves in chemical plants.

Gavin Taylor, US Naval Academy

Tom Adams, McMaster University


#To Train:
To train, run $WORKDIR/autovalves/learner/cstr_ex.py using the following command
mpiexec -np <numprocs> python cstr_ex.py <learning rate> <timestep> <entropy> <value coefficent> 

#To Evaluate:

#To see results:
In order to see the results from the evaluation, execute the 'qsub statFinder.sh' script in the $WORKDIR/autovalves directory.  This will execute python /autovalves/simulator/expSun31/extractData.py and will print out the stats from the evalution in the stdout file for statFinder.sh in autovalves.

In order to find three random runs within an hdf5 file and plot their states, execute python $WORKDIR/autovalves/learner/plotStates.py with a command line argument for the name of the file within the autovalves/learner/hdf5 directory.  The full path name is not needed; only add the filename as an argument.


