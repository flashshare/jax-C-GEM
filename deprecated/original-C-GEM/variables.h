/*____________________________________*/
/*variables.h 			      */
/*declare global variables, functions */
/*last modified: 24/01/08 sa             */
/*____________________________________*/

#ifndef VARIABLES_H
#define VARIABLES_H
#include "define.h"


/*_________init.c__________*/
extern void Init  ();
/*_________Ut.c__________*/
extern double Min(double, double);
extern double Max(double, double);

/*_________hyd.c__________*/
extern void Hyd(int);

/*_________bcforcing.c__________*/
extern double Tide(int);
extern double WS(int);
extern double windspeed(int, int);
extern double Discharge(int, int);
extern double Discharge_ups(int);
extern double Discharge_DN(int);
extern double Discharge_TT(int);
extern double Discharge_CANAL(int);
extern double Discharge_an(int);
extern double Discharge_VT(int);
extern void bgboundary(int);

/*_________tridaghyd.c__________*/
extern void Coeffa(int);
extern void Tridag();
extern double Conv(int,int,double, double*, double*);

/*_________transport.c__________*/
extern void Transport(int);

/*_________hydup.c__________*/
extern void Newbc(int);
extern void Update();
extern void NewUH();

/*_________transportup.c__________*/
extern void Dispcoef(int);
extern void Openbound(double*, int);
extern void TVD(double*, int);
extern void Disp(double*);
extern void Boundflux(int);

/*_________Ut.c__________*/
extern double Min(double, double);
extern double Max(double, double);

/*_________file.c_________*/
extern void Hydwrite(int);
extern void Transwrite(double*, char s[10], int);
extern void Rates(double*, char s[50], int);
extern void Fluxwrite(int, int);
extern void readFile(int, double*, double*, char s[50]);

/*_________biogeo.c_________*/
extern void Biogeo(int);

/*_________biogeout.c_______*/
extern double Tabs(int);
extern double waterT(int);
extern double I0(int);
extern double Pbmax(int, int);
extern double kmort(int,int);
extern double kmaintenance(int,int);
extern double Fhetox(int);
extern double Fhetden(int);
extern double Fnit(int);
extern double O2sat(int, int);
extern double Diff(int);
extern double Sc(int, int);
extern double KH(int, int);

/*__________Hyd.c____________________________*/

double Y[M+1];			/*convergence*/
double E[M+1];			/*convergence*/
double ZZ[M+1];			/*cross-section at reference level*/
double Chezy[M+1];      /*Chezy coefficient*/
double FRIC[M+1];		/*friction coefficient*/
double Mero[M+1];       /*erosion coefficient*/
double D[M+1];			/*total cross-section*/
double tau_ero[M+1];    /*critical shear stress for erosion*/
double tau_dep[M+1];    /*critical shear stress for deposition*/
double rs[M+1];			/*storage ratio variation*/

double Dold[M+1];
double PROF[M+1];		/*water depth*/
double slope[M+1];		/*bottom slope*/
double elevation[M+1];	/*water level = PROF - slope*/
double H[M+1];			/*free cross section*/
double U[M+1];			/*velocity*/
double B[M+1];			/*width*/
double TH[M+1];			/*temp free cross section*/
double TU[M+1];			/*temp velocity*/

double C[M+1][5];
double Z[M+1];

double Qdata[50000];
double Qtdata[50000];
double temdata[50000];
double temtdata[50000];
double Iodata[700000];
double Iotdata[700000];
double winddata[50000];
double windtdata[50000];

double toctub[50000];
double tocdataub[50000];
double O2tub [50000];
double O2dataub [50000];
double NH4tub [50000];
double NH4dataub [50000];
double NO3tub [50000];
double NO3dataub [50000];
double phy1tub [50000];
double phy1dataub [50000];
double phy2tub [50000];
double phy2dataub [50000];
double situb [50000];
double sidataub [50000];
double po4tub [50000];
double po4dataub [50000];
double dictub [50000];
double dicdataub [50000];
double talktub [50000];
double talkdataub [50000];
double spmtub [50000];
double spmdataub [50000];

double stlb [50000]; //xw:add Sal
double sdatalb [50000]; //xw:add Sal
double toctlb[50000];
double tocdatalb[50000];
double O2tlb [50000];
double O2datalb [50000];
double NH4tlb [50000];
double NH4datalb [50000];
double NO3tlb [50000];
double NO3datalb [50000];
double phy1tlb [50000];
double phy1datalb [50000];
double phy2tlb [50000];
double phy2datalb [50000];
double sitlb [50000];
double sidatalb [50000];
double po4tlb [50000];
double po4datalb [50000];
double dictlb [50000];
double dicdatalb [50000];
double talktlb [50000];
double talkdatalb [50000];
double spmtlb [50000];
double spmdatalb [50000];

double Htdata [50000];
double Hdata [50000];

double Qt_data_TT [50000];
double Q_data_TT [50000];
double toc_time_TT [50000];
double toc_data_TT [50000];
double O2_time_TT [50000];
double O2_data_TT [50000];
double NH4_time_TT [50000];
double NH4_data_TT [50000];
double NO3_time_TT [50000];
double NO3_data_TT [50000];
double si_time_TT [50000];
double si_data_TT [50000];
double po4_time_TT [50000];
double po4_data_TT [50000];
double phy1_time_TT [50000];
double phy1_data_TT [50000];
double phy2_time_TT [50000];
double phy2_data_TT [50000];
double dic_time_TT [50000];
double dic_data_TT [50000];
double talk_time_TT [50000];
double talk_data_TT [50000];
double spm_time_TT [50000];
double spm_data_TT [50000];

double Qt_data_DN [50000];
double Q_data_DN [50000];
double toc_time_DN [50000];
double toc_data_DN [50000];
double O2_time_DN [50000];
double O2_data_DN [50000];
double NH4_time_DN [50000];
double NH4_data_DN [50000];
double NO3_time_DN [50000];
double NO3_data_DN [50000];
double si_time_DN [50000];
double si_data_DN [50000];
double po4_time_DN [50000];
double po4_data_DN [50000];
double phy1_time_DN [50000];
double phy1_data_DN [50000];
double phy2_time_DN [50000];
double phy2_data_DN [50000];
double dic_time_DN [50000];
double dic_data_DN [50000];
double talk_time_DN [50000];
double talk_data_DN [50000];
double spm_time_DN [50000];
double spm_data_DN [50000];

double Qt_data_VT [50000];
double Q_data_VT [50000];
double toc_time_VT [50000];
double toc_data_VT [50000];
double O2_time_VT [50000];
double O2_data_VT [50000];
double NH4_time_VT [50000];
double NH4_data_VT [50000];
double NO3_time_VT [50000];
double NO3_data_VT [50000];
double si_time_VT [50000];
double si_data_VT [50000];
double po4_time_VT [50000];
double po4_data_VT [50000];
double phy1_time_VT [50000];
double phy1_data_VT [50000];
double phy2_time_VT [50000];
double phy2_data_VT [50000];
double dic_time_VT [50000];
double dic_data_VT [50000];
double talk_time_VT [50000];
double talk_data_VT [50000];
double spm_time_VT [50000];
double spm_data_VT [50000];

double Qt_data_CANAL [50000];
double Q_data_CANAL [50000];
double toc_time_CANAL [50000];
double toc_data_CANAL [50000];
double O2_time_CANAL [50000];
double O2_data_CANAL [50000];
double NH4_time_CANAL [50000];
double NH4_data_CANAL [50000];
double NO3_time_CANAL [50000];
double NO3_data_CANAL [50000];
double si_time_CANAL [50000];
double si_data_CANAL [50000];
double po4_time_CANAL [50000];
double po4_data_CANAL [50000];
double phy1_time_CANAL [50000];
double phy1_data_CANAL [50000];
double phy2_time_CANAL [50000];
double phy2_data_CANAL [50000];
double dic_time_CANAL [50000];
double dic_data_CANAL [50000];
double talk_time_CANAL [50000];
double talk_data_CANAL [50000];
double spm_time_CANAL [50000];
double spm_data_CANAL [50000];


/*__________Transport.c & Biogeo.c ____________________________*/

enum chem  {Phy1, Phy2, Si, NO3, NH4, PO4, PIP, O2, TOC, S, SPM, DIC, AT, HS, PH, AlkC, CO2};  /*species array; note: Phy1&Phy2 should be the first elements*/

#ifndef STRUCT_VERB
#define STRUCT_VERB
struct Verb {
  char   name [10];
  int    env;                    /*pelagic species=1, benthic species=0*/
  double c[M+1];                 /*concentration*/
  double clb;					 /*concentration at downstream boundary*/
  double cub;					 /*concentration at upstream boundary*/
  double avg[M+1];
  double concflux[M+1];			 /*c-flux*/
  double advflux[M+1];			/*advective flux*/
  double disflux[M+1];			/*dispersive flux*/
} v[MAXV];
#else
  extern struct Verb  v[MAXV];
#endif

#ifndef STRUCT_VERBTT
#define STRUCT_VERBTT
struct VerbTT {
  char   name [10];
  double c;                 /*concentration*/
} TT[MAXV];
#else
  extern struct VerbTT  TT[MAXV];
#endif

#ifndef STRUCT_VERBVT
#define STRUCT_VERBVT
struct VerbVT {
  char   name [10];
  double c;                 /*concentration*/
} VT[MAXV];
#else
  extern struct VerbVT  VT[MAXV];
#endif

#ifndef STRUCT_VERBCA
#define STRUCT_VERBCA
struct VerbCanal {
  char   name [10];
  double c;                 /*concentration*/
} Canal[MAXV];
#else
  extern struct VerbCanal  Canal[MAXV];
#endif

#ifndef STRUCT_VERBAN
#define STRUCT_VERBAN
struct VerbDN {
  char   name [10];
  double c;                 /*concentration*/
} DN[MAXV];
#else
  extern struct VerbDN  DN[MAXV];
#endif

double K;

double NPP_NO3[M+1][2];
double NPP_NO3_tot[M+1];
double NPP_NH4[M+1][2];
double NPP_NH4_tot[M+1];
double GPP[M+1][2];
double phydeath[M+1][2];
double phydeath_tot[M+1];
double NPP[M+1];
double Si_consumption[M+1];
double adegrad[M+1];
double denit[M+1];
double nitrif[M+1];
double o2air[M+1];
double sorption[M+1];
double erosion_s[M+1];
double deposition_s[M+1];
double erosion_v[M+1];
double deposition_v[M+1];
double integral[M+1];
double nlim[M+1];
double tau_b[M+1];
double Chezy[M+1];
double kwind[M+1];
double kflow[M+1];
double vp[M+1];
double co2air[M+1];
double reactionDIC[M+1];
double reactionTA[M+1];
double reactionHS[M+1];
double hco3[M+1];
double co3[M+1];
double pCO2[M+1];
double pCO2atmo;

int include_trib;
int include_WWTP;

double alpha[2];
double Pb[2];
double kmax[2];
double kexcr[2];
double kgrowth[2];
double kmaint[2];
double kmortality[2];
double KSi[2];
double KN[2];
double KPO4[2];
double KTOC;
double KO2_ox;
double KO2_nit;
double KNO3;
double KNH4;
double KinO2;
double redsi;
double redn;
double redp;
double kox;
double kdenit;
double knit;
double pCO2atm;
double kbg;
double kspm;
double Euler;
double ws;
double rho_w;
double g;

/*__________Transport.c____________________________*/

double fl[M+1];                  /*transport flux*/
double disp[M+1];                /*Dispersion coefficient*/
double watflux[M+1];			/*water flux*/

double AC_low;
double AC_up;
#endif


