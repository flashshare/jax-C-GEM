/*____________________________________*/
/*uptransport.c 		                  */
/*transport chemical species          */
/*last modified: 03/23 sa             */
/*____________________________________*/

#include "define.h"
#include "variables.h"


void Transport(int t)
{
 int s, i;

 Dispcoef(t);
 for(s=0; s<MAXV; s++)
 {
    if(v[s].env==1)
    {
      Openbound(v[s].c, s);
      TVD(v[s].c, s);
      Disp(v[s].c);
    }

    for(i=1; i<=M; i+=1)
    {
      v[s].avg[i]=v[s].avg[i]+v[s].c[i];
    }
    Boundflux(s);
     if((double)t/(double)(TS*DELTI)-floor((double)t/(double)(TS*DELTI))==0.0 && t>=WARMUP)
	{
		Transwrite(v[s].c, v[s].name, t);
		//Fluxwrite(s,t); //Use this for flux calculation, comment to save time// mol/s
	}
 }

}
