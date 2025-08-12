/*____________________________________*/
/*tridag.c 			                  */
/*calculate coefficients for          */
/*tridiagonal matrix, solve matrix,   */
/*iterate solution                    */
/*last modified: 03/07 sa             */
/*____________________________________*/

#include "define.h"
#include "variables.h"


 void Coeffa (int t)
 {
	int j,i;
    double Q_TT, Q_VT, Q_CANAL, Q_DN;

    Q_TT=Discharge_TT(t);
    Q_VT=Discharge_VT(t);
    Q_CANAL=Discharge_DN(t);
    Q_DN=Discharge_DN(t);

	for (j=3; j<=M3; j+=2)
	{
		i=j+1;
		Z[j]=RS*H[j]/DELTI;         /*RS=storage width ratio*/
		C[j][1]=-D[j-1]/(2.*DELXI);
        	C[j][2]=RS/DELTI;
        	C[j][3]=D[j+1]/(2.*DELXI);
        	C[j][4]=0.;

        	Z[i]=1./(G*DELTI)*U[i];
			C[i][1]=-1./(2.*DELXI*B[i-1]);
        	C[i][2]=1./(G*DELTI) + (FRIC[i]*fabs(U[i]))/PROF[i]+ (U[i+2]-U[i-2])/(4.*G*DELXI);
        	C[i][3]=1./(2.*DELXI*B[i+1]);
        	C[i][4]=0.;

            Z[31]=rs[31]*TH[31]/DELTI - Q_DN/(2.*DELXI); //to be commented if include_trib=0 in init.c Dongnai River -->deactive if no tributary
            Z[37]=rs[37]*TH[37]/DELTI - Q_CANAL/(2.*DELXI);
            Z[45]=rs[45]*TH[45]/DELTI - Q_VT/(2.*DELXI);
            Z[61]=rs[61]*TH[61]/DELTI - Q_TT/(2.*DELXI);
	}

	Z[2]=1./(2.*DELXI)*(H[1]/B[1]) + 1./(G*DELTI)*U[2];
        Z[M1]=RS*TH[M1]/DELTI - Discharge(t,i)/(2.*DELXI);

	C[2][1]=0.;
	    C[2][2]=1./(G*DELTI) + (FRIC[2]*fabs(U[2]))/PROF[2]+ (U[4]-U[2])/(2.*G*DELXI);
        C[2][3]=1./(2.*DELXI*B[3]);
        C[2][4]=0.;

	C[M1][1]=-D[M2]/(2.*DELXI);
        C[M1][2]=RS/DELTI;
        C[M1][3]=0.;
        C[M1][4]=0.;

}


void Tridag()
{
       double gam[M+1],var[M+1];
	   double bet;
       int j, i;


	   for(i=0;i<M+1;i++)
	   {
		    gam[i]=(00.0);
			var[i]=(00.0);
	   }

       bet   =C[2][2];
       var[2]=Z[2]/bet;

       for (j=3; j<=M1; j+=1)
        {
			if(j==M1)
				j=M1;
        	gam[j]=C[j-1][3]/bet;
        	bet   =C[j][2]-C[j][1]*gam[j];
        	var[j]=(Z[j]-C[j][1]*var[j-1])/bet;
	}

       for (j=M2; j>=2; j-=1)
        	var[j]=var[j]-gam[j+1]*var[j+1];

       for (j=2; j<=M2; j+=2)
       {
        	TU[j]   = var[j];
        	TH[j+1] = var[j+1];

       }
}



double Conv(int s, int e, double toler, double* xarray, double* yarray)
{
	int i,r;
	double diff, t;

	t=0.0;
	r=0.0;
  	diff=0.0;
	for(i=s; i<=e; i+=2)
	{
		diff = xarray[i] - yarray[i];
        	if(fabs(diff)>=t) t = fabs(diff);

	}

	r= fabs(t)>= toler ? 0.0 : 1.0;

	for(i=s; i<=e; i+=2)
  {
		yarray[i] = xarray[i];
	}
	return r;
}
