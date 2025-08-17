/*____________________________________*/
/*main.c 		            	      */
/*main time-loop                      */
/*Hyd: bcforcing, uphyd, tridaghyd    */
/*Transport: uptransport              */
/*Stuff: file, Ut                     */
/*____________________________________*/

#include "define.h"
#include "variables.h"
#include <time.h>


int main ()
{
     FILE *fptr1;
     long int t;
     clock_t start=clock();

     Init();

    for (t=0; t<=MAXT; t+=DELTI)
    {
		// printf("t:%f\n",(double)t/(24.*60.*60.));

        Hyd(t);
        bgboundary(t); //Load BC and lateral inputs
        Transport(t);
        if (t>WARMUP) //-3600*24*10
        {
            Biogeo(t);
        }
    }
    return (clock()-start)/CLOCKS_PER_SEC;
}


