/*____________________________________*/
/*init.c 			                  */
/*initialize. . . . . .               */
/*last modified: June 2021 AN         */
/*____________________________________*/

#include "define.h"
#include "variables.h"
#include "string.h"


void Init()
{
 int i,t,s;
 double Chezy_up;
 double Chezy_middle;
 double Chezy_low;
 double Mero_up;
 double Mero_mid;
 double Mero_low;
 double tau_up;
 double tau_mid;
 double tau_low;

//  delete old files from last simulation: Windows-compatible version
#ifdef _WIN32
    system("if exist OUT\\*.csv del /Q OUT\\*.csv");
    system("if exist OUT\\Flux\\*.csv del /Q OUT\\Flux\\*.csv");  
    system("if exist OUT\\Hydrodynamics\\*.csv del /Q OUT\\Hydrodynamics\\*.csv");
    system("if exist OUT\\Reaction\\*.csv del /Q OUT\\Reaction\\*.csv");
#else
    system("rm -f ./OUT/*.csv");
    system("rm -f ./OUT/Flux/*.csv");
    system("rm -f ./OUT/Hydrodynamics/*.csv");
    system("rm -f ./OUT/Reaction/*.csv");
#endif


//  include tributaries
    include_trib=1; //1 to include both tributaries; 0 to not

//	test setup
	fmod((double)M,2.) !=0 ? printf("Warning: %d \t Number of nodes M not even! Check EL:%d\t DELXI:%d\t and M:%d\n", M % 2,  EL, DELXI, M): printf("M=%d Ok!\n",M);


    readFile(50000, windtdata, winddata, "INPUT/Boundary/wind.csv");

    // Daily input of boundary condition. UB is freshwater input, LB is estuary mouth
    // Upstream BC: Q, TOC, DO, NH4, NO3, Diatom, Non-diatom, Silica, PO4, SPM
	readFile(50000, Qtdata, Qdata, "INPUT/Boundary/UB/discharge_ub.csv");
    readFile(50000, toctub, tocdataub, "INPUT/Boundary/UB/TOC_ub.csv");
    readFile(50000,O2tub,O2dataub,"INPUT/Boundary/UB/O2_ub.csv");
    readFile(50000,NH4tub,NH4dataub,"INPUT/Boundary/UB/NH4_ub.csv");
    readFile(50000,NO3tub,NO3dataub,"INPUT/Boundary/UB/NO3_ub.csv");
    readFile(50000,phy1tub,phy1dataub,"INPUT/Boundary/UB/dia_ub.csv");
    readFile(50000,phy2tub,phy2dataub,"INPUT/Boundary/UB/ndia_ub.csv");
    readFile(50000,situb,sidataub,"INPUT/Boundary/UB/si_ub.csv");
    readFile(50000,po4tub,po4dataub,"INPUT/Boundary/UB/po4_ub.csv");
    readFile(50000,spmtub,spmdataub,"INPUT/Boundary/UB/spm_ub.csv");

    //Dowstream BC: Elevation, T, (as upstream), Salinity
    readFile(50000,Htdata, Hdata, "INPUT/Boundary/LB/elevation.csv");
    readFile(50000,Iotdata, Iodata, "INPUT/Boundary/LB/Light.csv");
	readFile(50000,temtdata, temdata, "INPUT/Boundary/LB/T.csv");
    readFile(50000,phy1tlb,phy1datalb,"INPUT/Boundary/LB/dia_lb.csv");
    readFile(50000,phy2tlb,phy2datalb,"INPUT/Boundary/LB/ndia_lb.csv");
    readFile(50000,O2tlb,O2datalb,"INPUT/Boundary/LB/O2_lb.csv");
    readFile(50000,NH4tlb,NH4datalb,"INPUT/Boundary/LB/NH4_lb.csv");
    readFile(50000,NO3tlb,NO3datalb,"INPUT/Boundary/LB/NO3_lb.csv");
    readFile(50000,sitlb,sidatalb,"INPUT/Boundary/LB/si_lb.csv");
    readFile(50000,po4tlb,po4datalb,"INPUT/Boundary/LB/po4_lb.csv");
    readFile(50000,toctlb,tocdatalb,"INPUT/Boundary/LB/TOC_lb.csv");
    readFile(50000,spmtlb,spmdatalb,"INPUT/Boundary/LB/spm_lb.csv");
    readFile(50000,stlb,sdatalb,"INPUT/Boundary/LB/S_lb.csv");

    // Daily input of canals and tributaries


    // Dongnai River-cell 31
    readFile(4250,Qt_data_DN,Q_data_DN,"INPUT/Tributaries/Dongnai/discharge.csv");
    readFile(4250,toc_time_DN,toc_data_DN,"INPUT/Tributaries/Dongnai/toc_ub.csv");
    readFile(4250,O2_time_DN,O2_data_DN,"INPUT/Tributaries/Dongnai/O2_ub.csv");
    readFile(4250,NH4_time_DN,NH4_data_DN,"INPUT/Tributaries/Dongnai/NH4_ub.csv");
    readFile(4250,NO3_time_DN,NO3_data_DN,"INPUT/Tributaries/Dongnai/NO3_ub.csv");
    readFile(4250,phy1_time_DN,phy1_data_DN,"INPUT/Tributaries/Dongnai/dia_ub.csv");
    readFile(4250,phy2_time_DN,phy2_data_DN,"INPUT/Tributaries/Dongnai/ndia_ub.csv");
    readFile(4250,si_time_DN,si_data_DN,"INPUT/Tributaries/Dongnai/si_ub.csv");
    readFile(4250,po4_time_DN,po4_data_DN,"INPUT/Tributaries/Dongnai/po4_ub.csv");
    readFile(4250,spm_time_DN,spm_data_DN,"INPUT/Tributaries/Dongnai/SPM_ub.csv");


    // Canals-cell 37
    readFile(4250,Qt_data_CANAL,Q_data_CANAL,"INPUT/Tributaries/Canals/discharge.csv");
    readFile(4250,toc_time_CANAL,toc_data_CANAL,"INPUT/Tributaries/Canals/toc_ub.csv");
    readFile(4250,O2_time_CANAL,O2_data_CANAL,"INPUT/Tributaries/Canals/O2_ub.csv");
    readFile(4250,NH4_time_CANAL,NH4_data_CANAL,"INPUT/Tributaries/Canals/NH4_ub.csv");
    readFile(4250,NO3_time_CANAL,NO3_data_CANAL,"INPUT/Tributaries/Canals/NO3_ub.csv");
    readFile(4250,phy1_time_CANAL,phy1_data_CANAL,"INPUT/Tributaries/Canals/dia_ub.csv");
    readFile(4250,phy2_time_CANAL,phy2_data_CANAL,"INPUT/Tributaries/Canals/ndia_ub.csv");
    readFile(4250,si_time_CANAL,si_data_CANAL,"INPUT/Tributaries/Canals/si_ub.csv");
    readFile(4250,po4_time_CANAL,po4_data_CANAL,"INPUT/Tributaries/Canals/po4_ub.csv");
    readFile(4250,spm_time_CANAL,spm_data_CANAL,"INPUT/Tributaries/Canals/SPM_ub.csv");


    // Vam Thuat River (Tham Luong)-cell 45
    readFile(4250,Qt_data_VT,Q_data_VT,"INPUT/Tributaries/VamThuat/discharge.csv");
    readFile(4250,toc_time_VT,toc_data_VT,"INPUT/Tributaries/VamThuat/toc_ub.csv");
    readFile(4250,O2_time_VT,O2_data_VT,"INPUT/Tributaries/VamThuat/O2_ub.csv");
    readFile(4250,NH4_time_VT,NH4_data_VT,"INPUT/Tributaries/VamThuat/NH4_ub.csv");
    readFile(4250,NO3_time_VT,NO3_data_VT,"INPUT/Tributaries/VamThuat/NO3_ub.csv");
    readFile(4250,phy1_time_VT,phy1_data_VT,"INPUT/Tributaries/VamThuat/dia_ub.csv");
    readFile(4250,phy2_time_VT,phy2_data_VT,"INPUT/Tributaries/VamThuat/ndia_ub.csv");
    readFile(4250,si_time_VT,si_data_VT,"INPUT/Tributaries/VamThuat/si_ub.csv");
    readFile(4250,po4_time_VT,po4_data_VT,"INPUT/Tributaries/VamThuat/po4_ub.csv");
    readFile(4250,spm_time_VT,spm_data_VT,"INPUT/Tributaries/VamThuat/SPM_ub.csv");

    // Thi Tinh-cell 61
    readFile(4250,Qt_data_TT,Q_data_TT,"INPUT/Tributaries/ThiTinh/discharge.csv");
    readFile(4250,toc_time_TT,toc_data_TT,"INPUT/Tributaries/ThiTinh/toc_ub.csv");
    readFile(4250,O2_time_TT,O2_data_TT,"INPUT/Tributaries/ThiTinh/O2_ub.csv");
    readFile(4250,NH4_time_TT,NH4_data_TT,"INPUT/Tributaries/ThiTinh/NH4_ub.csv");
    readFile(4250,NO3_time_TT,NO3_data_TT,"INPUT/Tributaries/ThiTinh/NO3_ub.csv");
    readFile(4250,phy1_time_TT,phy1_data_TT,"INPUT/Tributaries/ThiTinh/dia_ub.csv");
    readFile(4250,phy2_time_TT,phy2_data_TT,"INPUT/Tributaries/ThiTinh/ndia_ub.csv");
    readFile(4250,si_time_TT,si_data_TT,"INPUT/Tributaries/ThiTinh/si_ub.csv");
    readFile(4250,po4_time_TT,po4_data_TT,"INPUT/Tributaries/ThiTinh/po4_ub.csv");
    readFile(4250,spm_time_TT,spm_data_TT,"INPUT/Tributaries/ThiTinh/SPM_ub.csv");

    //	Hydrodynamics: initialize arrays
//______________________________________________________________________________
    for (i=0; i<=M; i++)
    {
    E[i]=0.0;
    Y[i]=-5.0;

    Chezy_up=18;
    Chezy_middle=20;
    Chezy_low=50;



    if (i<=18){
        Chezy[i]= Chezy_low;
    }else{
        if (i<=35){
            Chezy[i]= Chezy_middle;
        }else{
            Chezy[i]= Chezy_up;
        }
    }

    FRIC[i]= 1.0/(Chezy[i]*Chezy[i]);

//erosion and deposition
    tau_up=0.9; // at upstream
    tau_mid=0.9;
    tau_low=0.45; //at mouth
    //tau_ero[i]= (i>=30) ? tau_low+(tau_up-tau_low)*(i-30)/(M-30) : tau_low;
    //tau_dep[i]= (i>=30) ? tau_low+(tau_up-tau_low)*(i-30)/(M-30) : tau_low;
        if (i<=31) {
            tau_ero[i]=0.45;
            tau_dep[i]=0.45;
        }else if (i<=85) {
            tau_ero[i]=0.6;
            tau_dep[i]=0.6;
        }else  {
            tau_ero[i]=0.8;
            tau_dep[i]=0.8;
        }

    Mero_up=9.0e-6; //kg/m2/s at Ben Suc, should be 1.e-5 if Mekong delta
    Mero_mid=3.0e-6;
    Mero_low=9.0e-6; // at Mouth, should be 1.e-6

        if (i<=31) {
            Mero[i]= 1.0e-5;
        }else if (i<=55) {
            Mero[i]= 1.0e-6;
        }else if (i<=75) {
            Mero[i]= 3.0e-6;
        }else  {
            Mero[i]= 2.0e-6;
        }

    U[i]=0.0;
    TU[i]=0.0;//temporary velocity

    //storage ratio
    if (i<=22) {
        rs[i]=1.0;
    }else if (i<=40) {
        rs[i]=1.0;
    }else if (i<=80) {
        rs[i]=1.0;
    }else { rs[i]=1.0;
    }

        if (i<=30){//width profile
            B[i]= B_Hon*(exp(-(i*((double)DELXI))/((double)LC_low)));
        }else{
            B[i]= B_infl*(exp(-((i-30)*((double)DELXI))/((double)LC_up)));
        }

        if (i<=11){
                    slope[i]= PROF0+(PROF1-PROF0)*(i-0)/(11-0); //separate to 3 parts
                }else if (i<=18){
                    slope[i]= PROF1+(PROF2-PROF1)*(i-11)/(18-11);
                }else if (i<=68){
                    slope[i]= PROF2;
                }else if (i<=79){
                    slope[i]= PROF2-(PROF2-8.35)*(i-68)/(79-68);
                }else{
                    slope[i]= 8.35-(8.35-3.49)*(i-79)/(M-79); //11 is remaining cell
                }

    ZZ[i]=B[i]*slope[i]; //cross-section at reference level=width x river depth (extract from bathymetry)
    Z[i]=0.0;
    for (t=0; t<=4; t++) C[i][t]=0.0;
    H[i] = B[i]*0.;
    TH[i]= B[i]*0.;
    D[i] =H[i]+ZZ[i];//cross-section of current water level = surface reference + surface by tide
    Dold[i] =H[i]+ZZ[i];
    PROF[i]=D[i]/B[i];

//	Transport: initialize arrays
//______________________________________________________________________________
    fl[i]=0.0;
    disp[i]=200.;
    watflux[i]=0.0;
    }


//  	Biogeochemistry: initialize structures
//______________________________________________________________________________
//Salinity [PSU]
	strcpy(v[S].name,"OUT/S.csv");
	v[S].env =1;			//1-transported
    v[S].cub =0.09;

	for (i=0; i<=M; i++)
	{
		v[S].c[i]       =v[S].clb+(v[S].cub-v[S].clb/((double)M))*i;
		v[S].avg[i]     =0.0;
		v[S].concflux[i]=0.0;
		v[S].advflux[i] =0.0;
		v[S].disflux[i] =0.0;
	}
//Siliceous Phytoplankton [mmol C/m3]
	strcpy(v[Phy1].name,"OUT/Dia.csv");
	v[Phy1].env =1;

	for (i=0; i<=M; i++)
	{
		v[Phy1].c[i]          =v[Phy1].clb+(v[Phy1].cub-v[Phy1].clb/((double)M))*i;
		v[Phy1].avg[i]     =0.0;
		v[Phy1].concflux[i]=0.0;
		v[Phy1].advflux[i] =0.0;
		v[Phy1].disflux[i] =0.0;
	}
//Non-siliceous Phytoplankton [mmol C/m3]
	strcpy(v[Phy2].name,"OUT/nDia.csv");
	v[Phy2].env =1;

	for (i=0; i<=M; i++)
	{
		v[Phy2].c[i]       =v[Phy2].clb+(v[Phy2].cub-v[Phy2].clb/((double)M))*i;
		v[Phy2].avg[i]     =0.0;
		v[Phy2].concflux[i]=0.0;
		v[Phy2].advflux[i] =0.0;
		v[Phy2].disflux[i] =0.0;
	}
//Silica [mmol Si/m3]
	strcpy(v[Si].name,"OUT/Si.csv");
	v[Si].env =1;


	for (i=0; i<=M; i++)
	{
		v[Si].c[i]        =v[Si].clb+(v[Si].cub-v[Si].clb/((double)M))*i;
		v[Si].avg[i]     =0.0;
		v[Si].concflux[i]=0.0;
		v[Si].advflux[i] =0.0;
		v[Si].disflux[i] =0.0;
	}
//Nitrate [mmol N/m3]
	strcpy(v[NO3].name,"OUT/NO3.csv");
	v[NO3].env =1;

	for (i=0; i<=M; i++)
	{
		v[NO3].c[i]       =v[NO3].clb+(v[NO3].cub-v[NO3].clb/((double)M))*i;
		v[NO3].avg[i]     =0.0;
		v[NO3].concflux[i]=0.0;
		v[NO3].advflux[i] =0.0;
		v[NO3].disflux[i] =0.0;
	}
//Ammonium [mmol N/m3]
	strcpy(v[NH4].name,"OUT/NH4.csv");
	v[NH4].env =1;

	for (i=0; i<=M; i++)
	{
		v[NH4].c[i]       =v[NH4].clb+(v[NH4].cub-v[NH4].clb/((double)M))*i;
		v[NH4].avg[i]     =0.0;
		v[NH4].concflux[i]=0.0;
		v[NH4].advflux[i] =0.0;
		v[NH4].disflux[i] =0.0;
	}
//Phosphate [mmol P/m3]
	strcpy(v[PO4].name,"OUT/PO4.csv");
	v[PO4].env =1;

	for (i=0; i<=M; i++)
	{
		v[PO4].c[i]         =v[PO4].clb+(v[PO4].cub-v[PO4].clb/((double)M))*i;
		v[PO4].avg[i]     =0.0;
		v[PO4].concflux[i]=0.0;
		v[PO4].advflux[i] =0.0;
		v[PO4].disflux[i] =0.0;
	}

//PIP Phosphate [mmol P/m3]
	strcpy(v[PIP].name,"OUT/PIP.csv");
	v[PIP].env =1;
    v[PIP].clb = 0.0328*1000/31;  	// from Nguyen TTN 2019, PIP=88% TPP of bed sediment --> TSS may be similar
	v[PIP].cub = 0.0164*1000/31;     // PIP (mg/L)= TSS (g/L)* TPP (mg/g)

	for (i=0; i<=M; i++)
	{
		v[PIP].c[i]         =v[PIP].clb-(v[PIP].clb-v[PIP].cub)/((double)M)*i;
		v[PIP].avg[i]     =0.0;
		v[PIP].concflux[i]=0.0;
		v[PIP].advflux[i] =0.0;
		v[PIP].disflux[i] =0.0;
	}

//Oxygen [mmol O/m3]
	strcpy(v[O2].name,"OUT/O2.csv");
	v[O2].env =1;

    for (i=0; i<=M; i++)
	{
		v[O2].c[i]        =v[O2].clb+(v[O2].cub-v[O2].clb/((double)M))*i;
		v[O2].avg[i]     =0.0;
		v[O2].concflux[i]=0.0;
		v[O2].advflux[i] =0.0;
		v[O2].disflux[i] =0.0;
	}
//Organic carbon [mmol C/m3]
	strcpy(v[TOC].name,"OUT/TOC.csv");
	v[TOC].env =1;

	for (i=0; i<=M; i++)
	{
		v[TOC].c[i]         =v[TOC].clb+(v[TOC].cub-v[TOC].clb/((double)M))*i;
		v[TOC].avg[i]     =0.0;
		v[TOC].concflux[i]=0.0;
		v[TOC].advflux[i] =0.0;
		v[TOC].disflux[i] =0.0;
	}
//Suspended matter [g/l]
	strcpy(v[SPM].name,"OUT/SPM.csv");
	v[SPM].env =1;

	for (i=0; i<=M; i++)
	{
		v[SPM].c[i]         =v[SPM].clb+(v[SPM].cub-v[SPM].clb/((double)M))*i;
		v[SPM].avg[i]     =0.0;
		v[SPM].concflux[i]=0.0;
		v[SPM].advflux[i] =0.0;
		v[SPM].disflux[i] =0.0;
	}


    //rate constants

for(i=0; i<=M; i++)
{
	for(s=Phy1; s<=Phy2; s++)
	{
		NPP_NO3[i][s]=0.0;
		NPP_NH4[i][s]=0.0;
		GPP[i][s]=0.0;
		phydeath[i][s]=0.0;
	}

	NPP[i]=0.0;
	Si_consumption[i]=0.0;
	sorption[i]=0.0;
	adegrad[i]=0.0;
	denit[i]=0.0;
	nitrif[i]=0.0;
	o2air[i]=0.0;
    co2air[i]=0.0;
}
Pb[Phy1]=5.58e-5;   // MAXIMUM SPECIFIC PHOTOSYNTHETIC RATE //Tref=20 C
Pb[Phy2]=5.58e-5;
alpha[Phy1]  =4.11e-7;   // PHOTOSYNTHETIC EFFICIENTY //Tref=20 C
alpha[Phy2]  =4.11e-7;
kexcr[Phy1]  =0.03; //EXCRETION CONSTANT //Tref=20 C
kexcr[Phy2]  =0.03;
kgrowth[Phy1]=0.3;  //GROWTH CONSTANT //Tref=20 C
kgrowth[Phy2]=0.3;
kmaint[Phy1] =1.6e-7;    // MAINTENANCE CONSTANT //Tref=20 C
kmaint[Phy2] =1.6e-7;
kmortality[Phy1]  =1.45e-7; // MORTALITY RATE CONSTANT //Tref=20 C
kmortality[Phy2]  =1.45e-7;
KSi[Phy1]    =7.07; //  HALF-SATURATION CONSTANT (Michaelis-Menten constant)
KSi[Phy2]    =0.0;
KN[Phy1]     =12.13; //   HALF-SATURATION CONSTANT (Michaelis-Menten constant)
KN[Phy2]     =12.13;
KPO4[Phy1]   =0.05;  //HALF-SATURATION CONSTANT (Michaelis-Menten constant)
KPO4[Phy2]   =0.05;
KTOC	     =312.5; //HALF-SATURATION CONSTANT (Michaelis-Menten constant)
KO2_ox		 =31; //HALF-SATURATION CONSTANT for aerobic degradation (Michaelis-Menten constant)
KO2_nit		 =51.25; //HALF-SATURATION CONSTANT for nitrification (Michaelis-Menten constant)
KinO2        =33.0; // INHIBITION CONTANT FOR DENITRIFICATION
KNO3	     =10.07; // HALF-SATURATION CONSTANT (Michaelis-Menten constant)
KNH4         =80.9;
redsi	     =15./106.;  //REDFIELD RATIO
redn	     =16./106.; //REDFIELD RATIO
redp         =1./106.;  //REDFIELD RATIO
kox			 =1.8e-4;   //AEROBIC DEGRADATION RATE CONSTANT //Tref=20 C
kdenit	     =3.05e-4;  //  DENITRIFICATION RATE CONSTANT //Tref=20 C
knit	     =1.6e-4; //  NITRIFICATION RATE CONSTANT //Tref=20 C
pCO2atmo     =390e-6;
kbg			 =1.3;      //1/m   BACKGROUND ATTENUATION - KD1. 1.3 kbg & 0 kspm increase much PHY
kspm         =0.001; //l/mg m    SPM ATTENUATION - KD2
Euler        =0.5772156649;
ws			 =1.e-4;       //m s-1 SETTLING VELOCITY
rho_w		 =1000.0;					//kg m-3
g            =9.82;						//m s-2
}
