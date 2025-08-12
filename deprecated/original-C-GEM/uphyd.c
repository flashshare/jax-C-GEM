#include "define.h"
#include "variables.h"


//set boundary value
void Newbc(int t)
{

	H[1]=B[1]*Tide(t);
	TH[1]=H[1];
	D[1]=H[1]+ZZ[1];
	PROF[1]=D[1]/B[1];
	U[M]=Discharge(t,M)/D[M];
	TU[M]=U[M];

}

//update TH, D, PROF, TU in iteration
void Update()
{
	int i;
	double tmp[M+1];

	for(i=2; i<=M2; i+=2)
	{
    		tmp[i-1]=TH[i-1]+ZZ[i-1];
    		tmp[i+1]=TH[i+1]+ZZ[i+1];
    		TH[i]=(TH[i-1]+TH[i+1])/2.0;
    		D[i]=(tmp[i-1]+tmp[i+1])/2.0;
    		PROF[i]=((tmp[i-1]/B[i-1])+(tmp[i+1]/B[i+1]))/2.0;
	}

	for(i=3; i<=M1; i+=2)
	{
		D[i]=TH[i]+ZZ[i];
		PROF[i]=D[i]/B[i];
		TU[i]=(TU[i+1]+TU[i-1])/2.0;
	}

	D[M]=(3.*(TH[M1]+ZZ[M1])-(TH[M3]+ZZ[M3]))/2.;
	PROF[M]=D[M]/B[M];
	TH[M]=(3.*TH[M1]-TH[M3])/2.;
	TU[1]=TU[2];

}

//update H & U
void NewUH()
{
	int i;

	for(i=1; i<=M; i++)
	{
		U[i]=TU[i];
		H[i]=TH[i];
		Dold[i]=D[i];
	}
}
