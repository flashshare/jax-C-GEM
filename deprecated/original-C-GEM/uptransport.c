/*____________________________________*/
/*uptransport.c 			              */
/*transport chemical species          */
/*last modified: 03/23 sa             */
/*____________________________________*/

#include "define.h"
#include "variables.h"

//calculate dispersion coefficients according to Van den Burgh with exp. cross section
void Dispcoef(int t)
{
	int i;
	double N,K,D0,beta,d,idis,AC,Bdis,Pdis;

	for (i=0; i<=M; i++)
	{
        if (i<=30){           // This needs to be consistent with with geometric set up in init.c, width profile
            AC=LC_low;
            Bdis=B_Hon;
            idis=1;
            Pdis = PROF0;}
        else{
            AC=LC_mid;
            Bdis=B_infl;
            idis=1;
            Pdis = PROF2;
        }
        K=4.38*pow(Pdis,0.36)*pow(Bdis,-0.21)*pow(AC,-0.14);
        N=-PI*(Discharge(t,i))/(Pdis*Bdis); //hydrodynamics number = (total river discharge during tidal period)/(saline water flow in estuary) at the mouth of estuary
        D0=26*pow(Pdis,1.5)*sqrt(N*G);
        beta=-(K*AC*Discharge(t,i))/(D0*B[i]*PROF[i]);
        d=D0*(1.- beta*( exp((i-idis)*DELXI/AC) -1.));
        disp[i]=(d>0.) ? d : 0.0;
	}

}

//set boundary conditions
void Openbound(double* co, int s)
{
  if(U[2]>=0.0)
    co[1]=co[1]- (co[1]-v[s].clb)*U[2]*((double)DELTI)/((double)DELXI);
      //co[1]=co[1]- (co[1]-v[s].clb)*U[2]*((double)DELTI)/(10000.); //to move lower bc further towards the sea
  else
    co[1]=co[1]- (co[3]-co[1])*U[2]*((double)DELTI)/((double)DELXI);

  if(U[M1]>=0.0)
    co[M]=co[M]- (co[M]-co[M2])*U[M1]*((double)DELTI)/((double)DELXI);
  else
    co[M]=co[M]- (v[s].cub-co[M])*U[M1]*((double)DELTI)/((double)DELXI);
    //co[M]=co[M]- (v[s].cub-co[M])*U[M1]*((double)DELTI)/(10000.); //to move lower bc further towards the sea

    co[M1]=co[M];
}

//advection scheme
void TVD (double* co, int s)
{
  int j;
  double vx, cfl, rg, philen,f, phi;
  double cold[M+1];

  for (j=1; j<=M; j++) cold[j]=co[j];


  for(j=1;j<=M2;j+=2)
  {
      vx=U[j];
      cfl =fabs(U[j])*((double)DELTI)/(2.*(double)DELXI);

//  For positive velocity, vx(j) > 0
      if(vx>0.0)
      {
          f = cold[j+2]-cold[j];
          if (fabs(f)>1.e-35 && j!=1)
          {
            rg = (cold[j]-cold[j-2])/f;
            phi= (2.-cfl)/3.+(1.+cfl)/3.*rg;
            philen = Max(0.,Min(2.,Min(2.*rg,phi)));
	        }
          else
          {
	          philen = 0.0;
          }
          co[j+1] = cold[j] + 0.5*(1.0-cfl)*philen*f;
          fl[j+1] = vx*D[j+1]*co[j+1];
      }

//  For negative velocity, vx(j) < 0

      if(vx<0.0)
      {
          f = cold[j]-cold[j+2];
          if (fabs(f)>1.e-35 && j!=M2)
          {
            rg = (cold[j+2]-cold[j+4])/f;
            phi= (2.-cfl)/3.+(1.+cfl)/3.*rg;
            philen = Max(0.,Min(2.,Min(2.*rg,phi)));
	        }
          else
          {
	          philen = 0.0;
          }
          co[j+1] = (cold[j+2] + 0.5*(1.0-cfl)*philen*f);
          fl[j+1] = vx*D[j+1]*co[j+1];
          }
      }

      for (j=3; j<=M2; j+=2)
      {
          co[j]=cold[j]-((double)DELTI)/(2.*(double)DELXI)*U[j]*(co[j+1]-co[j-1]);
      }

}

//dispersion scheme
void Disp(double* co)
{
  int i,j;
  double a[M+1], b[M+1], c[M+1], r[M+1], di[M+1], gam[M+1];
  double bet, c1, c2;

  a[1] =0.0;
  b[1] =1.;
  c[1] =0.;
  di[1]=co[1];

  a[M] =0.0;
  b[M] =1.;
  c[M] =0.;
  di[M]=co[M];

  for(i=2; i<=M1; i+=1)
  {
    c1=disp[i-1]*D[i-1]/D[i];
    c2=disp[i+1]*D[i+1]/D[i];
    a[i]=-c1*(double)DELTI/(2*(double)DELXI*(double)DELXI);
    c[i]=-c2*(double)DELTI/(2*(double)DELXI*(double)DELXI);
    b[i]=1.+c2*(double)DELTI/(2*(double)DELXI*(double)DELXI)+c1*(double)DELTI/(2*(double)DELXI*(double)DELXI);
    r[i]=1.-c2*(double)DELTI/(2*(double)DELXI*(double)DELXI)-c1*(double)DELTI/(2*(double)DELXI*(double)DELXI);
  }

  for(i=2; i<=M1; i+=1)
  {
    di[i]=-c[i]*co[i+1]+r[i]*co[i]-a[i]*co[i-1];
  }

  //Tridag
  if(b[1]==0.0)
  {
  	printf("Error Disp.c: b[1]==0");
	exit(EXIT_SUCCESS);
  }
  bet=b[1];
  co[1]=di[1]/bet;

  for(j=2; j<=M; j+=1)
  {
    gam[j]=c[j-1]/bet;
    bet=b[j]-a[j]*gam[j];
    co[j]=(di[j]-a[j]*co[j-1])/bet;
  }

  for(j=M-1; j>=2; j-=1)
  {
    co[j]=co[j]-gam[j+1]*co[j+1];
  }
}

// calculate fluxes through the boundaries
void Boundflux(int s)
{
  int j;
  double differ, disp_term;

//fl(j): flux at cell interface computed by tvd.f
//units: mol/l*m3/s
//c_even: concentration at cell interface (mol/l)
  for(j= 2; j<=M1; j+=2)
  {
    if(s==1) watflux[j] = watflux[j] + (fl[j]/v[s].c[j]); //m3/s
    v[s].advflux[j] = v[s].advflux[j] + fl[j]*DELTI;       //mol/l*m3/s*l/m3
    v[s].concflux[j]= v[s].concflux[j] + v[s].c[j];       // mol/l

    differ = (v[s].c[j+1] - v[s].c[j-1])*DELTI/((double)DELXI);          //mol/l*l/m3*1/m
    disp_term = differ*D[j]*disp[j];                          //mol/m4*m2*m2/s
    v[s].disflux[j] = v[s].disflux[j] + disp_term;
  }
}
