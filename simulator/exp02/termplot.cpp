/* ========================================================================== 

  Terminal Plotter
  Simple 2D terminal plotter
  
  NO INPUT VALIDATION
  
  Thomas A. Adams II
  
  tput cols
  tput lines
 ========================================================================== */
#include "termplot.h"
#include <math.h>
#include <vector>
#include <iostream>
#include <sys/ioctl.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

using namespace std;

// requires all options
int termplot(
  vector< vector<double> >& x, // inner vector has x0 x1
  vector<double>& x0bounds, // {min, max}   
  vector<double>& x1bounds, // {min, max}
  vector<double>& termsize // {rows, cols}
) {
 
  // test, get the terminal size
  
 
  int plotwidth; 
  int plotheight;
 
  int margin = 10;
  int N = x.size();
  
  if (2 == termsize.size()) {
	  plotwidth = termsize[1] - margin; 
    plotheight = termsize[0];
	} else {
	  struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
	  plotwidth = w.ws_col - margin; 
    plotheight = w.ws_row - margin;
	}
  

  vector<char> emptyrow(plotwidth, ' ');
  vector< vector<char> > plot(plotheight, emptyrow); // area to plot
  double x0range = (x0bounds[1] - x0bounds[0]);
  double x1range = (x1bounds[1] - x1bounds[0]);
  int c;
  int r;
  int i;
  
  // for each point in the vector (skip the first point, do that last so it's not overwritten)
  for (i=1; i<N; i++) {
	  // find the array index closest to the point
	  c = round( double(plotwidth)  * (x[i][0] - x0bounds[0])/x0range);
	  r = round( double(plotheight) * (x[i][1] - x1bounds[0])/x1range);
	  // if in the range to plot
		if (c>=0 && c < plotwidth && r>=0 && r < plotheight) {
		  if (0==i) { // first character
		    plot[r][c] = 'O';
		  } else if (N-1 == i) { // last point
			  plot[r][c] = 'X';  // other character
			} else {
			  plot[r][c] = '*';
			}
		}		 
	}
	
	// now do it for the initial point.
	c = round( double(plotwidth)  * (x[0][0] - x0bounds[0])/x0range);
	r = round( double(plotheight) * (x[0][1] - x1bounds[0])/x1range);
	if (c>=0 && c < plotwidth && r>=0 && r < plotheight) {
	  plot[r][c] = 'O';
	}
	
	// plot the plot
	
  
  char label[margin]; 
	double labelx;
	int len;
	int xpos;
	int target;
	for (r=0; r<plotheight; r++) {
	  
    
	
	  // border
	  if (0==r%5) {
	    labelx = (double(r)/double(plotheight)) * x1range + x1bounds[0];
	    sprintf(label, "%8g", labelx);  
	    label[9] = '\0';
		  cout << label;
		  for (i=0;i<margin-1-strlen(label);i++) {
			  cout << " ";
			}
		  
			cout << "+";
		} else {
		  
		  for (i=0;i<margin-1;i++) {
			  cout << " ";
			}
			cout << "|";
		}
	  for (c=0; c	<plotwidth; c++) {
	    cout << plot[r][c];
		}
		cout << endl;
	}
	
  for (i=0;i<margin-1;i++) {
	  cout << " ";
	}	
	// bottom border
	for (c=0; c	<plotwidth; c++) {
	  if (0==c%5) {
		  cout << "+";
		} else {
		  cout << "-";
		}  
	}
  cout << endl;  
    
    
  // bottom labels
  xpos = 1;
	for (c=0; c	<plotwidth; c++) {
	  if (0==c%10) {
	    
			labelx = (double(c)/double(plotwidth)) * x0range + x0bounds[0];
	    sprintf(label, "%10g", labelx);  
	    label[9] = '\0';
		  len = strlen(label);
		  target = margin + c + 1 - round(0.5*double(len));
		  if (target + len > margin + plotwidth) {
			  target = margin + plotwidth - len;
			} else if (target < 1) {
			  target = 1; 
			}
			
			while (xpos < target) {
			  cout << " ";
				xpos++;
			}
			
			cout << label;
			
			xpos += len;
			
		}  
	}
	
	cout << endl;  
  return(0); 
}
