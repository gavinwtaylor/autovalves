#include "gtSimEnv.h"

CSTREnv::CSTREnv():u0(2,0),numsteps(0),x0scale(0.45),x1scale(65) {
  xsp=N_VNew_Serial(2);
  x=N_VNew_Serial(2);
  abstol=N_VNew_Serial(2);
  NV_Ith_S(abstol, 0) = RCONST(1.0e-8);  // ok with CA (0...2) to the 1e-8      mol/m3
  NV_Ith_S(abstol, 1) = RCONST(1.0e-6);  // ok with T (300-400) to the 1e06
  reltol  = RCONST(1.0e-4); // ok with relative error of 0.01%
  cvode_mem =CVodeCreate(CV_BDF,CV_NEWTON);             // specify using backwards difference methods (stiff)
  CVodeInit(cvode_mem, cstrfun2, RCONST(0.0), x);  // initialize at time zero
  CVodeSVtolerances(cvode_mem, reltol, abstol);    // specify the tolerances
  SUNLinearSolver LS=SUNDenseLinearSolver(x, SUNDenseMatrix(2, 2));
  CVDlsSetLinearSolver(cvode_mem, LS, SUNDenseMatrix(2,2)); // specify the dense linear solver
  CVodeSetMaxNumSteps(cvode_mem, 5000);             // sets the maximum number of steps
  CVodeSetUserData(cvode_mem, &u0);                // sets the user data pointer
  // TEMP
  // execute for 1 minute

  reset();
}
boost::python::tuple CSTREnv::step(boost::python::tuple action){
  u0[0]=boost::python::extract<double>(action[0]);
  u0[1]=boost::python::extract<double>(action[1]);
  int flag = CVode(cvode_mem, t + tstep, x, &t, CV_NORMAL);
  double reward=calcReward();      
  bool done = false;
  if(!withinOval())
    done = true;
  rdat.push_back(reward);	
  done = steadyCheck();
  if (flag < 0) {
    cout << "CVode error"  << endl;
    done=true;
  }

  numsteps++;

  if(numsteps >= maxit)
    done = true;

  boost::python::tuple state=boost::python::make_tuple(NV_Ith_S(x,0),NV_Ith_S(x,1));
  boost::python::tuple retVal=boost::python::make_tuple(state,reward,done);
  return retVal;
}

double CSTREnv::calcReward(){
  double x0scaleinverse=1/x0scale;
  double x1scaleinverse=1/x1scale;
  double r=-(
      ((NV_Ith_S(x, 0) - NV_Ith_S(xsp, 0))*x0scaleinverse)*((NV_Ith_S(x, 0) - NV_Ith_S(xsp, 0))*x0scaleinverse) + 
      ((NV_Ith_S(x, 1) - NV_Ith_S(xsp, 1))*x1scaleinverse)*((NV_Ith_S(x, 1) - NV_Ith_S(xsp, 1))*x1scaleinverse)
      );
  return r;
}

void CSTREnv::reset(){
  tstep = RCONST(0.01); // the reporting interval / the time between updates to the control variabels.
  realtype tfin = RCONST(10); // the desired final time
  t = RCONST(0.000);
  maxit = tfin / tstep + RCONST(100);
  u0[0] = 0; 
  u0[1] = 0;   
  numsteps = 0; 
  rdat.clear(); 

  // initialize x in a circle surroudning the region of interestit** 
  // set rad to random number between 0 and 2 pi...would happen in initial set up and when you reset
  //TODO*rad = ((double)rand()/RAND_MAX) / (2.0 * M_PI);
  double rad = 2;
  NV_Ith_S(x, 0)   = 0.55 + x0scale * cos(rad); // mol/m3
  NV_Ith_S(x, 1)   = 375 + x1scale * sin(rad); // deg K

  // new setpoint
  // STANDARD SETPOINT
  NV_Ith_S(xsp, 0)   = 0.57; // mol/m3
  NV_Ith_S(xsp, 1)   = 395.3; // deg K

  // reinitialize the integrator --> **reset**
  CVodeReInit(cvode_mem, RCONST(0.0),x);
}

CSTREnv::~CSTREnv(){
  N_VDestroy_Serial(x);
  N_VDestroy_Serial(xsp);
  N_VDestroy_Serial(abstol);

  // Free integrator memory 
  CVodeFree(&cvode_mem);
}

static int cstrfun2(realtype t, N_Vector x, N_Vector xp, void *user_data) {
  // recast the user data pointer
  vector<double>* u = static_cast< vector<double>* >(user_data); 

  // Precalculate some common terms.
  realtype intermed = k0 * NV_Ith_S(x,0) * exp( -E / (R * NV_Ith_S(x,1)) );

  // CA' ==> since #defines are used, repeated calculations of F/V and cp*rho are done once during compiling so this takes no extra flops
  NV_Ith_S(xp,0) = (F/V) * ( (*u)[0] - NV_Ith_S(x,0)) - intermed; 

  // T'
  NV_Ith_S(xp,1) = (F/V) * (TIN - NV_Ith_S(x,1)) + ( DH/(cp * rho) ) * intermed + (*u)[1] / (cp * rho * V);

  return(0);
}

bool CSTREnv::steadyCheck(){
  if (numsteps >= REWARDCHECK) {
    double rewardsum = 0;
    for (int j=rdat.size() - 1; j >= rdat.size() - REWARDCHECK; j--)
      rewardsum = rewardsum + fabs(rdat[j]);
    // stop if our reward is smaller than our cumulative tolerance over three iterations
    if (REWARDTOL > rewardsum){
      cout << "YES! STEADY STATE"<<endl;
      return true;
    }
  }
  return false;
}
bool CSTREnv::withinOval(){
  double centerx0=(NV_Ith_S(x,0)-.55)/x0scale;
  double centerx1=(NV_Ith_S(x,1)-375)/x1scale;
  double dist=centerx0*centerx0+centerx1*centerx1;
  return dist<1;
}
/*
int main(void) {
  int k;
  //MPI_Init(NULL,NULL);
  //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  // casual variables
  clock_t clockin, clockout;

  // allocate memory for results storage and provide initial variables
  vector<double> tdat;  // initial state and main memory object for t
  vector<double> x0(2, 0); // empty vector  
  vector< vector<double> > udat; // u state space. start with zeros [CAin, Q].
  vector< vector<double> > xdat;  // x state space history (reporting intervals)
  vector<double> rdat; // reward data
  vector<double> u0(2, 0); // the current control action.
  vector<double> clearvec(2,0); // something convenient

  // steady state checkers
  int j;  // storage for temporary sums used later
  double rewardsum;
  int rewardcheck = 10; // how many timesteps to check?
  double reward=0.0;
  double rewardtol = 0.0001; // the reward tolerance for when we've reached our goal. 
  // Since it's scaled on the rnage, a reward tolernace of 0.0001 means within 0.01% of the range

  // this is our desired setpoint
  N_Vector xsp     = N_VNew_Serial(2);

  // CV ODE types for use with the integrator. Initial states and settings.
  N_Vector x       = N_VNew_Serial(2); // create an x vector of two state variables

  N_Vector abstol  = N_VNew_Serial(2); // the vector of absolute tolerances
  NV_Ith_S(abstol, 0) = RCONST(1.0e-8);  // ok with CA (0...2) to the 1e-8      mol/m3
  NV_Ith_S(abstol, 1) = RCONST(1.0e-6);  // ok with T (300-400) to the 1e06

  realtype reltol  = RCONST(1.0e-4); // ok with relative error of 0.01%

  // Create the CV ODE memory space and provide settings.
  void *cvode_mem;
  cvode_mem =CVodeCreate(CV_BDF,CV_NEWTON);             // specify using backwards difference methods (stiff)
  CVodeInit(cvode_mem, cstrfun2, RCONST(0.0), x);  // initialize at time zero
  CVodeSVtolerances(cvode_mem, reltol, abstol);    // specify the tolerances
  SUNLinearSolver LS=SUNDenseLinearSolver(x, SUNDenseMatrix(2, 2));
  CVDlsSetLinearSolver(cvode_mem, LS, SUNDenseMatrix(2,2)); // specify the dense linear solver
  CVodeSetMaxNumSteps(cvode_mem, 5000);             // sets the maximum number of steps
  CVodeSetUserData(cvode_mem, &u0);                // sets the user data pointer
  //cout << "our memory " << &u0 << endl;

  // TEMP
  // execute for 1 minute
  realtype tstep = RCONST(0.01); // the reporting interval / the time between updates to the control variabels.
  realtype tfin = RCONST(10); // the desired final time
  realtype t; // the time at the end of the integrator, which may be earlier than tout if it failed
  realtype maxit = tfin / tstep + RCONST(100);

  // start the timer
  clockin = clock();
  double rad = 0;

  double x0scale = 0.45;
  double x1scale = 65;
  double x0scaleinverse = 1/x0scale;
  double x1scaleinverse = 1/x1scale;

  int runswitherrors = 0;

  // reset variables to beginning 
  t = RCONST(0.000);
  xdat.clear();
  tdat.clear();
  rdat.clear();
  udat.clear();

  //reset(&u0, x, xsp, &rdat, &i, &rad, x0scale, x1scale, cvode_mem, &reward);
  rdat.push_back(reward);

  double foo[8] = {NV_Ith_S(x, 0), NV_Ith_S(x,1), reward, 0, NV_Ith_S(xsp,0), NV_Ith_S(xsp,1), x0scale, x1scale };


  while (true) { // a little safety check on max iterations.
    //get msg - if exit, call exit and break, if reset, call reset, else:
    // execute the ODE for one control step
    double done = 0;
    //MPI_Status status;
    double env_data[4]; //action and state info coming from the environment
    double ret_data[4]; //state, reward, and done status being returned to env
    //MPI_Recv(env_data, 4, MPI_DOUBLE, partner, MPI_ANY_TAG, MPI_COMM_WORLD,&status);
    u0[0] = env_data[0];
    u0[1] = env_data[1]; 
    NV_Ith_S(x,0) = env_data[2];
    NV_Ith_S(x,1)=env_data[3];
    //if(status.MPI_TAG == 2){ //exit 
    if (true){
      cleanUp(x, abstol, cvode_mem);
      return 0;
    }
    //else if(status.MPI_TAG == 1){ //reset
    else if (true){
      //reset(&u0, x, xsp,&rdat,&i,&rad,x0scale,x1scale,cvode_mem,&reward);
    }
    //else if(status.MPI_TAG == 3){ //init
    else if (true){
      //MPI_Send(foo, 8, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD);
      continue;
    }
    else{ //action	
      int flag = CVode(cvode_mem, t + tstep, x, &t, CV_NORMAL);

      //reward=calcReward(x,xsp,x0scaleinverse,x1scaleinverse);      
      if(!withinOval(x, x0scale, x1scale)){
        reward=reward*100;
        done = 1;
      }

      rdat.push_back(reward);	
      //done = steadyCheck(rdat,rewardcheck,rewardtol,i);
      if (flag < 0) {
        cout << "CVode error"  << endl;
        done=1;
      }

      i++;

      if(i >= maxit){
        done = 1;
      }
    }

    ret_data[0] = NV_Ith_S(x,0);
    ret_data[1] = NV_Ith_S(x,1);
    ret_data[2] = reward;
    ret_data[3] = done;
    //cout<<"About to send new state from simulator "<< state[0]<< " "<<state[1]<<endl;
    //MPI_Send(ret_data, 4, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD);

  }

  return(0);
  }





  void cleanUp(N_Vector& x, N_Vector& abstol, void* cvode_mem) {
    // Free state space and abstol vectors 
    N_VDestroy_Serial(x);
    N_VDestroy_Serial(abstol);

    // Free integrator memory 
    CVodeFree(&cvode_mem);
  }


  // THIS IS THE MAIN REACATOR FUNCTION
  // State variables are x[0] = CA in kmol/m3
  // x[1] is T in K
  // data passed in via void* are the current controller outputs
  // u[0] = CAIn in kmol/m3
  // u[1] = Q in kJ/min
  */

using namespace boost::python;
BOOST_PYTHON_MODULE(cstr){
  class_<CSTREnv>("CSTREnv")
    .def("reset", &CSTREnv::reset)
    .def("step", &CSTREnv::step)
    ;
}
