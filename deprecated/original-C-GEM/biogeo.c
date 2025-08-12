/*----------------------------------------*/
/*  biogeo.c 			                  */
/*  reaction network                      */
/*  last modified: 27/09/2021 - Set DSi = 0 if NPP consume all DSi                        */
/*----------------------------------------*/

#include "define.h"
#include "variables.h"
#include<math.h>

//_____Dissociation constant of CO2 (mmol m-3) < RTM : Cai & Wang
double disK1 (double S, double t)
{
    double T=waterT(t)+273.15;
	double pK1 = -14.8425+(3404.71/T)+(0.032786*T);
	double f1 = -0.0230848-(14.3456/T);
	double f2 = 0.000691881+(0.429955/T);
	return pow(10,-(pK1+(f1*pow(S,0.5))+(f2*S)));
}

//_____Dissociation constant of CO2 (mmol m-3) <RTM : Cai & Wang
double disK2(double S, double t)
{
    double T=waterT(t)+273.15;
    double pK2 = -6.4980+(2902.39/T)+(0.02379*T);
	double f3 = -0.458898+(41.24048/T);
	double f4 = 0.0284743-(2.55895/T);
	return pow(10,-(pK2+(f3*pow(S,0.5))+(f4*S)));
}

//_____Dissociation constant of water (mmol m-3) <RTM : Cai & Wang
double disK3(double S, double t)
{
    double T=waterT(t)+273.15;
    double lnKw = (-13847.26/T)+(148.9652)-(23.6521*log(T));
    double f = (118.67/T)-(5.977)+(1.0495*log(T));
	double g = -0.01615;
	return exp(lnKw+(f*sqrt(S))+(g*S));
}

//_____Dissociation constant of sulfide (mmol m-3) :  Dickson 1990b
double disK4(double S, double t)
{
	double lnK4,lnK4sansunite, T,I;
    T=waterT(t)+273.15;
	I=(19.919*S)/(1000-1.00198*S);
	lnK4=(-4276.1/T)+141.328-(23.093*log(T));
	lnK4sansunite=(((-13856/T)+324.57-47.986*log(T))*pow(I,0.5)) + (((35474/T)-771.54+114.723*log(T))*I) - ((2698/T)*pow(I,1.5)) + ((1776/T)*pow(I,2)) + lnK4;
    return exp(lnK4sansunite+lnK4);
}

//_____Dissociation constant of boron (mmol m-3) : Dickson 1990a (idem Sandra)
double disK5(double S, double t)
{
	double Ts, lnK5, KK5;
    Ts=waterT(t)+273.15;
	lnK5=(-8966.90-2890.53*sqrt(S)-77.942*S+1.728*pow(S,1.5)-0.0996*S*S)/Ts+(148.0248+137.1942*sqrt(S)+1.62142*S)+((-24.4344-25.085*sqrt(S)-0.2474*S)*log(Ts))+ (0.053105*sqrt(S)*Ts);
	KK5=exp(lnK5);
	return(KK5);
}
void Biogeo(int t)
{
	double nlim1,nlim2,KD, Ebottom, psurf, pbot;  //, tau_b;
	double appGAMMAsurf, appGAMMAbot;
    double Q_ups, Q_TT, Q_VT, Q_CANAL, Q_DN;
	double nswitch;
	int i, s;
	FILE *fptr1;

    double temp, alk_carb, tb, H, newh,K1,K2,K3,K4,K5,temperature;
    double xi1,xi2,beta1,gamma1;

    double sulf=0;
    double Pac, Kps;
/*---------------------------------------*/
/* Compute reaction rates [mmol/(m3 s)]  */
/*---------------------------------------*/

	for(i=1; i<=M; i++)
	{
	    //avoid uselessly low values of S
		if (v[S].c[i]<1.0e-10)
		    v[S].c[i]=0;

		for(s=Phy1; s<=Phy2; s++)
		{

			KD = kbg + kspm *(v[SPM].c[i] + 12.0/35.0/0.7 * v[Phy1].c[i] + 12.0/35.0/0.3 * v[Phy2].c[i]);
			Ebottom = I0(t) * exp(-KD * PROF[i]);
            psurf = I0(t) * alpha[s] / Pbmax(t,s);
			pbot = Ebottom * alpha[s] / Pbmax(t,s);

            if(psurf<=1)
				appGAMMAsurf = -(log(psurf) + Euler - psurf + pow(psurf,2.) / 4. - pow(psurf,3.) / 18. + pow(psurf,4.) / 96. - pow(psurf,5.) / 600.);
			else
				appGAMMAsurf = exp(-psurf) * (1. / (psurf + 1. - (1. / (psurf + 3. - (4. / (psurf + 5. - (9. / (psurf + 7. - (16. / (psurf + 9.))))))))));

			if(pbot<=1)
                appGAMMAbot = -(log(pbot) + Euler + pbot - pow(pbot,2.)/4. + pow(pbot,3.)/18. - pow(pbot,4.)/96. + pow(pbot,5.)/600.);
			else
				appGAMMAbot = exp(-pbot) * (1. / (pbot + 1. - (1. / (pbot + 3. - (4. / (pbot +5. - (9. / (pbot +  7. - (16. / (pbot + 9.))))))))));


			integral[i] = (I0(t) <= 0)||(Ebottom<1e-300) ? 0.0 : (appGAMMAsurf - appGAMMAbot + log(I0(t) / Ebottom)) / (KD);
			nlim[i] = v[Si].c[i] /(v[Si].c[i] + KSi[s]) * (v[NH4].c[i]) / (v[NH4].c[i] + KN[s]) * (v[PO4].c[i] / (v[PO4].c[i] + KPO4[s]));
			//nlim should remove NO3 because NO3 only consume when all NH4 is consumed by phytoplankton
			//Si also need to remove if there is no diatoms
			//nlim = v[Si].c[i] /(v[Si].c[i] + KSi[s]) * (v[NO3].c[i] + v[NH4].c[i]) / ((v[NH4].c[i] + v[NO3].c[i]) + KN[s]) * (v[PO4].c[i] / (v[PO4].c[i] + KPO4[s]));
			nswitch = v[NH4].c[i] / (10. + v[NH4].c[i]);
			GPP[i][s] = (Pbmax(t,s) * v[s].c[i] * nlim[i] * integral[i]); //[mmol C m-2 s-1]
			NPP_NO3[i][s] = ((1. - nswitch) * ((GPP[i][s]/PROF2)) * (1. - kexcr[s]) * (1. - kgrowth[s]) - kmaintenance(t,s) * v[s].c[i] >0)  ? (1. - nswitch) * ((GPP[i][s]/PROF2)) * (1. - kexcr[s]) * (1. - kgrowth[s]) - kmaintenance(t,s) * v[s].c[i] : 0;
			NPP_NH4[i][s] = (nswitch * ((GPP[i][s]/PROF2)) * (1. - kexcr[s]) * (1. - kgrowth[s]) - kmaintenance(t,s) * v[s].c[i] > 0 ) ? nswitch * ((GPP[i][s]/PROF2)) * (1. - kexcr[s]) * (1. - kgrowth[s]) - kmaintenance(t,s) * v[s].c[i]  : 0;
			phydeath[i][s] = kmort(t,s) * v[s].c[i];

		}

            kflow[i]=pow((fabs(U[i]))*Diff(t)/PROF[i],0.5);
			kwind[i]=(1/3.6e5)*0.31*(windspeed(t,i)*windspeed(t,i))*pow(Sc(t,i)/660,-0.5);
			vp[i]=kflow[i]+kwind[i];


			tau_b[i]=rho_w*g*U[i]*U[i]/(Chezy[i]*Chezy[i]);
			erosion_s[i]= (tau_ero[i] >= tau_b[i])	?	0.0 : Mero[i]*(tau_b[i]/tau_ero[i]-1.0);                     //surface erosion flux
			deposition_s[i]=(tau_dep[i] >= tau_b[i])	?	ws*(1.0-tau_b[i]/tau_dep[i])*v[SPM].c[i] : 0.0;          //surface deposition flux

			erosion_v[i]=(1.0/PROF[i])*erosion_s[i];                                                                 //volume erosion flux
			deposition_v[i]=(1.0/PROF[i])*deposition_s[i];                                                           //volume deposition flux

/*-------------------------------------------------------------*/
/*            Biogeochemical rates [mmol m-3 s-1]              */
/*-------------------------------------------------------------*/
        NPP_NO3_tot[i] = NPP_NO3[i][Phy1]+NPP_NO3[i][Phy2];
        NPP_NH4_tot[i] = NPP_NH4[i][Phy1]+NPP_NH4[i][Phy2];
        phydeath_tot[i] = phydeath[i][Phy1]+phydeath[i][Phy2];

			NPP[i] = NPP_NO3[i][Phy1] + NPP_NO3[i][Phy2] + NPP_NH4[i][Phy1] + NPP_NH4[i][Phy2]; //mmol C

			Si_consumption[i] = Min(0.0,-redsi*NPP[i]); //mmol Si
            adegrad[i] = (Fhetox(t) * v[TOC].c[i] / (v[TOC].c[i]+KTOC) * v[O2].c[i] / (v[O2].c[i] + KO2_ox));  //mmol C
            if(isnan(adegrad[i]))
                printf("Error! Check reaction module\n");
			denit[i] = (Fhetden(t) * v[TOC].c[i] / (v[TOC].c[i] + KTOC) * KinO2 / (v[O2].c[i] + KinO2) * v[NO3].c[i] / (v[NO3].c[i] + KNO3)); //mmol C
			nitrif[i] = (Fnit(t) * v[O2].c[i] / (v[O2].c[i] + KO2_nit) * v[NH4].c[i] / (v[NH4].c[i] + KNH4));    //mmol N
            o2air[i] = (vp[i]/PROF[i])*(O2sat(t,i) - v[O2].c[i]);   //mmol O

            //SRP adsorption
            if (i<=30)
            {
                Pac=2.580/31; //mmol/g
                Kps=0.02*1000/31; //mmol/m3
            } else if (i<=57)
            {
                Pac=2.580/31;
                Kps=0.02*1000/31;
            } else {
                Pac=2.580/31;
                Kps=0.02*1000/31;}

            sorption[i]=Pac*v[SPM].c[i]*1000*v[PO4].c[i]/(v[PO4].c[i]+Kps); //calculate the adsorbed PIP potential by Langmuir isotherm, it includes the current PIP and sorption
            sorption[i]=sorption[i]- v[PIP].c[i]; // update PIP by subtract the real PIP in the river, if it is > 0 , it means that there is adsorption process which converts PO4 (or SRP) to PIP, if <0 is the desorption
            sorption[i]=sorption[i]/(DELTI+10*60); //convert sorption to kinetic mmolP/m3/s, 60*60 means that after 3600s the sorption process reach
            //sorption[i] = 0; //for testing
	}


/*--------------------------------------------------*/
/* Update biogeochemical state variables [mmol/m3]  */
/*--------------------------------------------------*/

    Q_ups=Discharge_ups(t);
    Q_DN=Discharge_DN(t);
    Q_TT=Discharge_TT(t);
    Q_VT=Discharge_VT(t);
    Q_CANAL=Discharge_CANAL(t);

	for (i=0; i<=M; i++)
	{
        if (include_trib==1 && i==31&& U[i]<0)
        {
            v[Phy1].c[i] =((v[Phy1].c[i+1]*Discharge(t,i+1)+DN[Phy1].c*Q_DN)/Discharge(t,i))+(NPP_NO3[i][Phy1]+NPP_NH4[i][Phy1]-phydeath[i][Phy1])*DELTI;
        }else if (include_trib==1 && i==37&& U[i]<0)
        {
            v[Phy1].c[i] =v[Phy1].c[i] - (Canal[Phy1].c-v[Phy1].c[i])*Q_CANAL*DELTI/(B[i]*DELXI*PROF[i])+(NPP_NO3[i][Phy1]+NPP_NH4[i][Phy1]-phydeath[i][Phy1])*DELTI;
        }else if (include_trib==1 && i==45&& U[i]<0)
        {
            v[Phy1].c[i] =v[Phy1].c[i] - (VT[Phy1].c-v[Phy1].c[i])*Q_VT*DELTI/(B[i]*DELXI*PROF[i])+(NPP_NO3[i][Phy1]+NPP_NH4[i][Phy1]-phydeath[i][Phy1])*DELTI;
        }else if (include_trib==1 && i==61&& U[i]<0)
        {
            v[Phy1].c[i] =v[Phy1].c[i] - (TT[Phy1].c-v[Phy1].c[i])*Q_TT*DELTI/(B[i]*DELXI*PROF[i]) +(NPP_NO3[i][Phy1]+NPP_NH4[i][Phy1]-phydeath[i][Phy1])*DELTI;
        }else{
            v[Phy1].c[i] =v[Phy1].c[i]+(NPP_NO3[i][Phy1]+NPP_NH4[i][Phy1]-phydeath[i][Phy1])*DELTI;
        }
        v[Phy1].c[i]= (v[Phy1].c[i] <= 0)	?	1e-5 : v[Phy1].c[i];

        if (include_trib==1 && i==31&& U[i]<0)
        {
            v[Phy2].c[i] =((v[Phy2].c[i+1]*Discharge(t,i+1)+DN[Phy2].c*Q_DN)/Discharge(t,i))+(NPP_NO3[i][Phy2]+NPP_NH4[i][Phy2]-phydeath[i][Phy2])*DELTI;
        }else if (include_trib==1 && i==37&& U[i]<0)
        {
            v[Phy2].c[i] =v[Phy2].c[i] - (Canal[Phy2].c-v[Phy2].c[i])*Q_CANAL*DELTI/(B[i]*DELXI*PROF[i])+(NPP_NO3[i][Phy2]+NPP_NH4[i][Phy2]-phydeath[i][Phy2])*DELTI;
        }else if (include_trib==1 && i==45&& U[i]<0)
        {
            v[Phy2].c[i] =v[Phy2].c[i] - (VT[Phy2].c-v[Phy2].c[i])*Q_VT*DELTI/(B[i]*DELXI*PROF[i])+(NPP_NO3[i][Phy2]+NPP_NH4[i][Phy2]-phydeath[i][Phy2])*DELTI;
        }else if (include_trib==1 && i==61&& U[i]<0)
        {
            v[Phy2].c[i] =v[Phy2].c[i] - (TT[Phy2].c-v[Phy2].c[i])*Q_TT*DELTI/(B[i]*DELXI*PROF[i]) +(NPP_NO3[i][Phy2]+NPP_NH4[i][Phy2]-phydeath[i][Phy2])*DELTI;
        }else{
            v[Phy2].c[i] =v[Phy2].c[i]+(NPP_NO3[i][Phy2]+NPP_NH4[i][Phy2]-phydeath[i][Phy2])*DELTI;
        }
        v[Phy2].c[i]= (v[Phy2].c[i] <= 0)	?	1e-5 : v[Phy2].c[i];


        if (include_trib==1 && i==31&& U[i]<0)
        {
            v[Si].c[i]  =((v[Si].c[i+1]*Discharge(t,i+1)+DN[Si].c*Q_DN)/Discharge(t,i)) + Min(0.0,-redsi*NPP[i])*DELTI+redsi*phydeath[i][Phy1]*DELTI;
        }else if (include_trib==1 && i==37&& U[i]<0)
        {
            v[Si].c[i]  =((v[Si].c[i+1]*Discharge(t,i+1)+Canal[Si].c*Q_CANAL)/Discharge(t,i)) + Min(0.0,-redsi*NPP[i])*DELTI+redsi*phydeath[i][Phy1]*DELTI;
        }else if (include_trib==1 && i==45&& U[i]<0)
        {
            v[Si].c[i]  =((v[Si].c[i+1]*Discharge(t,i+1)+VT[Si].c*Q_VT)/Discharge(t,i)) + Min(0.0,-redsi*NPP[i])*DELTI+redsi*phydeath[i][Phy1]*DELTI;
        }else if (include_trib==1 && i==61&& U[i]<0)
        {
            v[Si].c[i]  =((v[Si].c[i+1]*Discharge(t,i+1)+TT[Si].c*Q_TT)/Discharge(t,i)) + Min(0.0,-redsi*NPP[i])*DELTI+redsi*phydeath[i][Phy1]*DELTI;
        }else{
            v[Si].c[i]  =v[Si].c[i] + Min(0.0,-redsi*NPP[i])*DELTI+redsi*phydeath[i][Phy1]*DELTI;
        }
        v[Si].c[i]= (v[Si].c[i] <= 0)	?	1e-5 : v[Si].c[i];

        if (include_trib==1 && i==31&& U[i]<0)
        {
            v[PO4].c[i] =v[PO4].c[i]-(DN[PO4].c-v[PO4].c[i])*Q_DN*DELTI/(B[i]*DELXI*PROF[i]) +(redp*(adegrad[i]+denit[i]-NPP[i])-sorption[i])*DELTI;
        }else if (include_trib==1 && i==37&& U[i]<0)
        {
            v[PO4].c[i] =v[PO4].c[i]-(Canal[PO4].c-v[PO4].c[i])*Q_CANAL*DELTI/(B[i]*DELXI*PROF[i]) +(redp*(adegrad[i]+denit[i]-NPP[i])-sorption[i])*DELTI;
        }else if (include_trib==1 && i==45&& U[i]<0)
        {
            v[PO4].c[i] =v[PO4].c[i]-(VT[PO4].c-v[PO4].c[i])*Q_VT*DELTI/(B[i]*DELXI*PROF[i]) +(redp*(adegrad[i]+denit[i]-NPP[i])-sorption[i])*DELTI;
        }else if (include_trib==1 && i==61&& U[i]<0)
        {
            v[PO4].c[i] =v[PO4].c[i]-(TT[PO4].c-v[PO4].c[i])*Q_TT*DELTI/(B[i]*DELXI*PROF[i]) +(redp*(adegrad[i]+denit[i]-NPP[i])-sorption[i])*DELTI;
        }else{
            v[PO4].c[i] =v[PO4].c[i] +(redp*(adegrad[i]+denit[i]-NPP[i])-sorption[i])*DELTI;
        }
        v[PO4].c[i]= (v[PO4].c[i] <= 0)	?	1e-5 : v[PO4].c[i];

        v[PIP].c[i]= v[PIP].c[i]+sorption[i]*DELTI;
        v[PIP].c[i]= (v[PIP].c[i] <= 0)	?	1e-5 : v[PIP].c[i];

        if (include_trib==1 && i==31&& U[i]<0)
        {
            v[O2].c[i] =v[O2].c[i] - (DN[O2].c-v[O2].c[i])*Q_DN*DELTI/(B[i]*DELXI*PROF[i])+(-adegrad[i]+NPP_NH4[i][Phy1]+NPP_NH4[i][Phy2]+(138./106.)*(NPP_NO3[i][Phy1]+NPP_NO3[i][Phy2])-2.*nitrif[i] + o2air[i])*DELTI;
        }else if (include_trib==1 && i==37&& U[i]<0)
        {
            v[O2].c[i] =v[O2].c[i] - (Canal[O2].c-v[O2].c[i])*Q_CANAL*DELTI/(B[i]*DELXI*PROF[i])+(-adegrad[i]+NPP_NH4[i][Phy1]+NPP_NH4[i][Phy2]+(138./106.)*(NPP_NO3[i][Phy1]+NPP_NO3[i][Phy2])-2.*nitrif[i] + o2air[i])*DELTI;
        }else if (include_trib==1 && i==45&& U[i]<0)
        {
            v[O2].c[i] =v[O2].c[i] - (VT[O2].c-v[O2].c[i])*Q_VT*DELTI/(B[i]*DELXI*PROF[i])+(-adegrad[i]+NPP_NH4[i][Phy1]+NPP_NH4[i][Phy2]+(138./106.)*(NPP_NO3[i][Phy1]+NPP_NO3[i][Phy2])-2.*nitrif[i] + o2air[i])*DELTI;
        }else if (include_trib==1 && i==61&& U[i]<0)
        {
            v[O2].c[i] =v[O2].c[i] - (TT[O2].c-v[O2].c[i])*Q_TT*DELTI/(B[i]*DELXI*PROF[i]) +(-adegrad[i]+NPP_NH4[i][Phy1]+NPP_NH4[i][Phy2]+(138./106.)*(NPP_NO3[i][Phy1]+NPP_NO3[i][Phy2])-2.*nitrif[i] + o2air[i])*DELTI;
        }else{
            v[O2].c[i] =v[O2].c[i] +(-adegrad[i]+NPP_NH4[i][Phy1]+NPP_NH4[i][Phy2]+(138./106.)*(NPP_NO3[i][Phy1]+NPP_NO3[i][Phy2])-2.*nitrif[i] + o2air[i])*DELTI;
        }
        v[O2].c[i]= (v[O2].c[i] <= 0)	?	1e-5 : v[O2].c[i];

        if (include_trib==1 && i==31&& U[i]<0)
        {
            v[TOC].c[i] =v[TOC].c[i] - (DN[TOC].c-v[TOC].c[i])*Q_DN*DELTI/(B[i]*DELXI*PROF[i])+(-adegrad[i]-denit[i]+phydeath_tot[i])*DELTI;
       }else if (include_trib==1 && i==37&& U[i]<0)
        {
            v[TOC].c[i] =((v[TOC].c[i+1]*Discharge(t,i+1)+Canal[TOC].c*Q_CANAL)/Discharge(t,i)) +(-adegrad[i]-denit[i]+phydeath_tot[i])*DELTI;
        }else if (include_trib==1 && i==45&& U[i]<0)
        {
            v[TOC].c[i] =((v[TOC].c[i+1]*Discharge(t,i+1)+VT[TOC].c*Q_VT)/Discharge(t,i)) +(-adegrad[i]-denit[i]+phydeath_tot[i])*DELTI;
        }else if (include_trib==1 && i==61&& U[i]<0)
        {
            v[TOC].c[i] =v[TOC].c[i] - (TT[TOC].c-v[TOC].c[i])*Q_TT*DELTI/(B[i]*DELXI*PROF[i]) +(-adegrad[i]-denit[i]+phydeath_tot[i])*DELTI;
        }else{
            v[TOC].c[i] =v[TOC].c[i] +(-adegrad[i]-denit[i]+phydeath_tot[i])*DELTI;
        }
        v[TOC].c[i]= (v[TOC].c[i] <= 0)	?	1e-5 : v[TOC].c[i];

        if (include_trib==1 && i==31&& U[i]<0)
        {
            v[NH4].c[i] =v[NH4].c[i] - (DN[NH4].c-v[NH4].c[i])*Q_DN*DELTI/(B[i]*DELXI*PROF[i]) +(redn*(adegrad[i]-NPP_NH4_tot[i])-nitrif[i])*DELTI;
        }else if (include_trib==1 && i==37&& U[i]<0)
        {
            v[NH4].c[i] =((v[NH4].c[i+1]*Discharge(t,i+1)+Canal[NH4].c*Q_CANAL)/Discharge(t,i))+(redn*(adegrad[i]-NPP_NH4_tot[i])-nitrif[i])*DELTI;
        }else if (include_trib==1 && i==45&& U[i]<0)
        {
            v[NH4].c[i] =((v[NH4].c[i+1]*Discharge(t,i+1)+VT[NH4].c*Q_VT)/Discharge(t,i))+(redn*(adegrad[i]-NPP_NH4_tot[i])-nitrif[i])*DELTI;
        }else if (include_trib==1 && i==61&& U[i]<0)
        {
            v[NH4].c[i] =((v[NH4].c[i+1]*Discharge(t,i+1)+TT[NH4].c*Q_TT)/Discharge(t,i)) +(redn*(adegrad[i]-NPP_NH4_tot[i])-nitrif[i])*DELTI;
        }else{
            v[NH4].c[i] =v[NH4].c[i] +(redn*(adegrad[i]-NPP_NH4_tot[i])-nitrif[i])*DELTI;
        }
        v[NH4].c[i]= (v[NH4].c[i] <= 0)	?	1e-5 : v[NH4].c[i];

        if (include_trib==1 && i==31&& U[i]<0)
        {
            v[NO3].c[i] =v[NO3].c[i] - (DN[NO3].c-v[NO3].c[i])*Q_DN*DELTI/(B[i]*DELXI*PROF[i])+(-94.4/106.*denit[i]+nitrif[i]-redn*NPP_NO3_tot[i])*DELTI;
        }else if (include_trib==1 && i==37&& U[i]<0)
        {
            v[NO3].c[i] =v[NO3].c[i] - (Canal[NO3].c-v[NO3].c[i])*Q_CANAL*DELTI/(B[i]*DELXI*PROF[i])+(-94.4/106.*denit[i]+nitrif[i]-redn*NPP_NO3_tot[i])*DELTI;
        }else if (include_trib==1 && i==45&& U[i]<0)
        {
            v[NO3].c[i] =v[NO3].c[i] - (VT[NO3].c-v[NO3].c[i])*Q_VT*DELTI/(B[i]*DELXI*PROF[i])+(-94.4/106.*denit[i]+nitrif[i]-redn*NPP_NO3_tot[i])*DELTI;
        }else if (include_trib==1 && i==61&& U[i]<0)
        {
            v[NO3].c[i] =v[NO3].c[i] - (TT[NO3].c-v[NO3].c[i])*Q_TT*DELTI/(B[i]*DELXI*PROF[i]) +(-94.4/106.*denit[i]+nitrif[i]-redn*NPP_NO3_tot[i])*DELTI;
        }else{
            v[NO3].c[i] =v[NO3].c[i] +(-94.4/106.*denit[i]+nitrif[i]-redn*NPP_NO3_tot[i])*DELTI;
        }
        v[NO3].c[i]= (v[NO3].c[i] <= 0)	?	1e-5 : v[NO3].c[i];


        if (include_trib==1 && i==31&& U[i]<0)
        {
            v[SPM].c[i] =v[SPM].c[i] - (DN[SPM].c-v[SPM].c[i])*Q_DN*DELTI/(B[i]*DELXI*PROF[i])+ 1.0/PROF[i]*(erosion_s[i]-deposition_s[i])*DELTI;
        }else if (include_trib==1 && i==37&& U[i]<0)
        {
            v[SPM].c[i] =v[SPM].c[i] - (Canal[SPM].c-v[SPM].c[i])*Q_CANAL*DELTI/(B[i]*DELXI*PROF[i])+ 1.0/PROF[i]*(erosion_s[i]-deposition_s[i])*DELTI;
        }else if (include_trib==1 && i==45&& U[i]<0)
        {
            v[SPM].c[i] =v[SPM].c[i] - (VT[SPM].c-v[SPM].c[i])*Q_VT*DELTI/(B[i]*DELXI*PROF[i])+ 1.0/PROF[i]*(erosion_s[i]-deposition_s[i])*DELTI;
        }else if (include_trib==1 && i==61&& U[i]<0)
        {
            v[SPM].c[i] =v[SPM].c[i] - (TT[SPM].c-v[SPM].c[i])*Q_TT*DELTI/(B[i]*DELXI*PROF[i])+ 1.0/PROF[i]*(erosion_s[i]-deposition_s[i])*DELTI;
        }else{
            v[SPM].c[i] =v[SPM].c[i] + 1.0/PROF[i]*(erosion_s[i]-deposition_s[i])*DELTI;
        }
        v[SPM].c[i]= (v[SPM].c[i] <= 0)	?	1e-5 : v[SPM].c[i];

    }
	//Write rates; concentrations are written to file in the transport routine
    if((double)t/(double)(TS*DELTI)-floor((double)t/(double)(TS*DELTI))==0.0 && t>=WARMUP)
	{
        Rates(NPP, "OUT/Reaction/NPP.csv", t);
		Rates(Si_consumption,"OUT/Reaction/Si_consumption.csv", t);
		Rates(NPP_NO3_tot, "OUT/Reaction/NPP_NO3.csv", t);
		Rates(NPP_NH4_tot, "OUT/Reaction/NPP_NH4.csv", t);
		Rates(phydeath_tot,"OUT/Reaction/phydeath.csv",t);
	    Rates(adegrad, "OUT/Reaction/adegrad.csv", t);
	    Rates(denit, "OUT/Reaction/denit.csv", t);
	    Rates(nitrif, "OUT/Reaction/nitrif.csv", t);
	    Rates(o2air, "OUT/Reaction/o2air.csv", t);
	    Rates(sorption, "OUT/Reaction/sorption.csv", t);
	    Rates(erosion_s, "OUT/Reaction/eross.csv", t);
	    Rates(deposition_s, "OUT/Reaction/deps.csv", t);
	    Rates(integral, "OUT/Reaction/integral.csv", t);
	    Rates(nlim, "OUT/Reaction/nlim.csv", t);

	    fptr1=fopen("OUT/Q_ups.csv","a");
	    fprintf(fptr1,"%f\n",Q_ups);
	    fclose(fptr1);

	}
}
