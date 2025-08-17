/*____________________________________*/
/*define.h			                  */
/*define global variables             */
/*last modified: June/2021 AN         */
/*____________________________________*/

#ifndef DEFINE_H
#define DEFINE_H

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAXT (50)*24*60*60   /* hydrodynamic & transport warm up + real simulation/saving data*/
#define WARMUP 30*24*60*60   /*warm up period; switch on saving function of biogeo, transport, hydro if t>WARMUP*/
#define DELTI 180           /*delta t [sec]*/
#define TS	20			/*save every TS. time step   (1--> every 180 seconds; 120 = 6 hours --> 4 values/day)*/
#define DELXI 2000          /*delta x [m]*/
#define EL    202000      /*estuarine length [m]*/ //NOTE!!!: the number of grid points M must be EVEN number

#define PROF0   9.61        /*depth at the estuarine mouth default 9.61[m]*/
#define PROF1   12.54        /*depth at the cell 12= 14.0[m]*/
#define PROF2   17.75        /*depth at the cell 18= 21.0[m]*/

#define B_Hon    3887           /*width at the mouth [m]*/
#define B_mid    1887           /*width at cell 12= [m]*/
#define B_infl    450           /*width at cell 31*/ //optics 400 bigger B increase Sal, very sensitive here

#define LC_low   65500         /*convergence length in mouth-the low estuarine zone 160000 [m],effect SPM*/
#define LC_mid   122500           /*convergence length cell 12*/ //10500 optics
#define LC_up    122500          /*convergence length at cell 31 [m] optic 62500, smaller LC gives smaller PO4 at upstream*/

#define G     9.81          /*gravity acceleration*/
#define TOL   1e-6          /*convergence criterium - FIXED: was 1e-10 (too strict)*/
#define MAXITS 1000         /*max number of iteration steps - FIXED: was 1000000 (too many)*/
#define EPS   0.00001
#define RS    1.0		    /*storage width ratio*/

#define AMPL  4.43          /*tidal amplitude at the month; average annual tidal amplitude in Vung Tau*/

#define PI	acos(-1.)       /*pi number 3.1416...*/
#define M	EL/DELXI+1      /*max grid points (MUST BE EVEN!)*/
#define M1	M-1	            /*max-1 grid points*/
#define M2	M-2	            /*max-2 grid point*/
#define M3	M-3	            /*max-3 grid point*/

#define MAXV 17             /*max number of species in chem array, the original C-GEM has 16 variables, this version add PIP for the adsorption process*/

#endif