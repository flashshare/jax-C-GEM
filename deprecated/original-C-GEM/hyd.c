/*____________________________________*/
/*hyd.c                               */
/*main hydrodynamic routine           */
/*last modified: 03/07 sa             */
/*____________________________________*/

#include "define.h"
#include "variables.h"


void Hyd(int t)
{
 int i;
 double rsum;
  Newbc(t);                   				        //set new boundary conditions
  rsum=0;
  i=0;

  do
  {
    i=i+1 ;
    Coeffa(t);              				        //set coefficient matrix
    Tridag();               				        //solve tridiagonal matrix
    rsum=Conv(3,M1,TOL,TH,E)+Conv(2,M2,TOL,TU,Y);  	//test convergence
    Update();               				        //update D, PROF
  } while(rsum!=2.);

  NewUH();                  				        //update U,H; set TH, TU=0
    if((double)t/(double)(TS*DELTI)-floor((double)t/(double)(TS*DELTI))==0.0 && t>=WARMUP)
	{
		Hydwrite(t);						        //write file

	}
}




