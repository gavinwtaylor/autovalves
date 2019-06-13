#ifndef GTSIMENV_H
#define GTSIMENV_H
/* ========================================================================== 

   Experiment 1
   Toy Problem with Linear MPC
   Initialized a million times in a ring around the desired steady state
   ========================================================================== */
#include <unistd.h>
#include <iostream>                                  
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <sstream>
#include <stdio.h>
#include <boost/python.hpp>
using namespace std;
namespace bp = boost::python;

// Header files for CVODE
#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <cvode/cvode_direct.h>       /* prototype for CVDense */
#include <sundials/sundials_dense.h> /* definitions DlsMat DENSE_ELEM */
#include <sundials/sundials_types.h> /* definition of type realtype */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */

static int cstrfun2(realtype t, N_Vector x, N_Vector xp, void *user_data);

#define REWARDTOL 0.0001
#define REWARDCHECK 10
  // define model constants 
#define F    RCONST(0.1)     // m3/min
#define V    RCONST(0.1)     // m3
#define TIN  RCONST(310.0)   // K
#define DH   RCONST(-4.78e4) // kJ/kmol
#define k0   RCONST(72e9)    // 1/min
#define E    RCONST(8.314e4) // kJ/kmol
#define cp   RCONST(0.239)   // kJ / kg K
#define rho  RCONST(1000.0)   // kg/m3
#define R    RCONST(8.314)   // kJ/kmol-K
class CSTREnv {
  private:
    void* cvode_mem;
    N_Vector x;
    N_Vector xsp;
    N_Vector abstol;
    void* user_data;
    realtype reltol;
    realtype maxit;
    vector<double> u0;
    int numsteps;
    double x0scale;
    double x1scale;
    vector<double> rdat;
    double rad;
    
    double calcReward();
    bool steadyCheck();
    bool withinOval();

  public:
    CSTREnv();
    ~CSTREnv();
    void reset();

};
// prototype for the function we have
#endif
