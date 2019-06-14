# autovalves

For an RL approach to autonomously control valves in chemical plants.

Gavin Taylor, US Naval Academy

Tom Adams, McMaster University


#To Train:
To train, run $WORKDIR/autovalves/learner/cstr_ex.py using the following command
mpiexec -np <numprocs> python cstr_ex.py <learning rate> <timestep> <entropy> <value coefficent> <num layers> <layer width> <num environments>

To start several training jobs, run python $WORKDIR/autovalves/simulator/expSun31/runtests.py. This will start several jobs with various combinations of hyperparameters.

Log files will be saved in $WORKDIR/autovalves/learner/logs
Models will be saved in $WORKDIR/autovalves/learner/models 

#To Evaluate:
To evaluate, run $WORKDIR/autovalves/learner/cstr_evaluate.py <number of trajectories>  

To start several evaluation jobs, do qsub $WORKDIR/autovalves/simulator/expSun31/runEvaluate.sh

The hdf5 files will be saved in $WORKDIR/autovalves/learner/hdf5

#To see results:
