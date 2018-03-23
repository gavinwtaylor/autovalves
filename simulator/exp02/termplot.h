/* ========================================================================== 

  Terminal Plotter
  Simple 2D terminal plotter
  
  NO INPUT VALIDATION
  
  Thomas A. Adams II
  
  tput cols
  tput lines
 ========================================================================== */
#include <vector>
using namespace std;

// requires all options
int termplot(
  vector< vector<double> >& x, // inner vector has x0 x1
  vector<double>& x0bounds, // {min, max}   
  vector<double>& x1bounds, // {min, max}
  vector<double>& termsize // {rows, cols}
);
