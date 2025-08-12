/*____________________________________*/
/*biogeout.c 			              */
/*forcing/functions for biogeo.c      */
/*last modified: 19/01/08 sa          */
/*____________________________________*/

#include "define.h"
#include "variables.h"

//Temperature
double waterT(int t)
{
	double day, restday, waterT;
	int nday;

	day=floor((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0);
	restday=((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0)-day;
	nday=(int)day;

	if(day<0.0)
	{
		waterT=temdata[nday];
	}
	else
	{
		waterT=(temdata[nday]+(temdata[nday+1]-temdata[nday])*restday); //linear interpolation of data
	}

	return waterT;
}


double I0(int t)  // Transient version
{
    double I, hour, resthour;
	int nhour;

	hour=floor((double)t/(60.0*60.0))-((double)WARMUP)/(60.0*60.0);
	resthour=((double)t/(60.0*60.0))-((double)WARMUP)/(60.0*60.0)-hour;
	nhour=(int)hour;

    if(hour<0.0)
    {
        I=Iodata[nhour];
    }
    else
    {
        I=Iodata[nhour]+(Iodata[nhour+1]-Iodata[nhour])*resthour; //linear interpolation of data
    }

	return I;
}
//Maximum photosynthetic production [1/s] - TEMPERATURE FUNCTION
double Pbmax(int t, int s)
{
    double P,a,PP;
    a=waterT(t)-20;
    PP=(pow(1.067,a));
    P=Pb[s]*PP;
    if(s==Phy2)	{P=P;}

	return P;
}

// T-dependence mortality constant [1/s] - TEMPERATURE FUNCTION
double kmort(int t, int s)
{
    double KMOR;
    KMOR=kmortality[s]*exp(0.07*waterT(t));
    if(s==Phy2) {KMOR=KMOR;}

    return KMOR;
}

// T-dependence maintenance constant [1/s] - TEMPERATURE FUNCTION
double kmaintenance(int t, int s)
{
    double KMAIN;

    KMAIN=kmaint[s]*exp(0.0322*(waterT(t)-20.));
    if(s==Phy2) {KMAIN=KMAIN;}

    return KMAIN;
}

//T-dependence aerobic degradation [mmolC/m3.s] - TEMPERATURE FUNCTION
double Fhetox(int t)
{
    double Fhet_OX, a, f;
    a=(waterT(t)-20)/10;
    f=pow(2,a);
    Fhet_OX=kox*f;
    return Fhet_OX;

}

//T-dependence denitrification [mmolC/m3.s] - TEMPERATURE FUNCTION
double Fhetden(int t)
{
    double Fhet_DEN, a, f;
    a=(waterT(t)-20);
    f=pow(1.07,a);
    Fhet_DEN=kdenit*f;

    return Fhet_DEN;

}

//T-dependence nitrification [mmolN/m3.s] - TEMPERATURE FUNCTION
double Fnit(int t)
{
    double F_NIT, a, f;
    a=(waterT(t)-20);
    f=pow(1.08,a);
//    f=pow(1.12,a);
    F_NIT=knit*f;

    return F_NIT;
}


//O2 saturation
double O2sat(int t, int i)
{
	double O2, T, f, lnO2sat;

    T=waterT(t)+273.15;
	lnO2sat=-1.3529996*100.+157228.8/T - 66371490./(T*T) + 12436780000./(T*T*T) - 8621061.*100000./(T*T*T*T);
	f=-0.020573 + 12.142/T -2363.1/(T*T);

	O2=exp(lnO2sat+f*v[S].c[i]);

	return O2;
}
//Molecular diffusion coefficient for O2
double Diff(int t)
{
    double Diff, T;

    //T=Tabs(t);
    T=waterT(t)+273.15;

    Diff=(6.35*T-1664.)*1.0e-11;

    return Diff;
}

//Schmidt number
double Sc(int t, int i)
{
    double Sc0, Sc;

    Sc0=1800.6-120.1*waterT(t)+3.7818*(waterT(t)*waterT(t))-0.047608*(waterT(t)*waterT(t)*waterT(t));

    Sc=Sc0*(1+(3.14e-3)*v[S].c[i]);


    return Sc;
}
