/*____________________________________*/
/*file.c                  		      */
/*write files                         */
/*last modified: 03/07 sa             */
/*____________________________________*/

#include "define.h"
#include "variables.h"



void Hydwrite (int t)
{
  FILE *fptr1;
  FILE *fptr2;
  FILE *fptr3;
  FILE *fptr4;
  FILE *fptr5;
  FILE *fptr6;
  FILE *fptr7;
  FILE *fptr12;
  FILE *fptr13;
  FILE *fptr14;
  FILE *fptr15;
  FILE *fptr16;

  int i;


  fptr1=fopen("OUT/Hydrodynamics/U.csv","a");
  fptr2=fopen("OUT/Hydrodynamics/H.csv","a");
  fptr3=fopen("OUT/Hydrodynamics/PROF.csv","a");
  fptr4=fopen("OUT/Hydrodynamics/tau_b.csv","a");
  fptr5=fopen("OUT/Hydrodynamics/B.csv","a");
  fptr6=fopen("OUT/Hydrodynamics/Chezy.csv","a");
  fptr7=fopen("OUT/Hydrodynamics/FRIC.csv","a");
  fptr12=fopen("OUT/Hydrodynamics/disp.csv","a");
  fptr13=fopen("OUT/Hydrodynamics/windspeed.csv","a");
  fptr14=fopen("OUT/Hydrodynamics/slope.csv","a");
  fptr15=fopen("OUT/Hydrodynamics/surface.csv","a");
  fptr16=fopen("OUT/Hydrodynamics/elevation.csv","a");

  fprintf(fptr1,"%d,",t);
  fprintf(fptr2,"%d,",t);
  fprintf(fptr3,"%d,",t);
  fprintf(fptr4,"%d,",t);
  fprintf(fptr5,"%d,",t);
  fprintf(fptr6,"%d,",t);
  fprintf(fptr7,"%d,",t);
  fprintf(fptr12,"%d,",t);
  fprintf(fptr13,"%d,",t);
  fprintf(fptr14,"%d,",t);
  fprintf(fptr15,"%d,",t);
  fprintf(fptr16,"%d,",t);

  for(i=0; i<=M; i++)
  {
    fprintf(fptr1,"%f,",U[i]);
    fprintf(fptr2,"%f,",H[i]);
    fprintf(fptr3,"%f,",PROF[i]);
    fprintf(fptr4,"%f,",tau_b[i]);
    fprintf(fptr5,"%f,",B[i]);
    fprintf(fptr6,"%f,",Chezy[i]);
    fprintf(fptr7,"%f,",FRIC[i]);
    fprintf(fptr12,"%f,",disp[i]);
    fprintf(fptr13,"%f,",windspeed(t,i));
    fprintf(fptr14,"%f,",slope[i]);
    fprintf(fptr15,"%f,",D[i]);
    fprintf(fptr16,"%f,",PROF[i]-slope[i]);
  }

  fprintf(fptr1,"\n");
  fprintf(fptr2,"\n");
  fprintf(fptr3,"\n");
  fprintf(fptr4,"\n");
  fprintf(fptr5,"\n");
  fprintf(fptr6,"\n");
  fprintf(fptr7,"\n");
  fprintf(fptr12,"\n");
  fprintf(fptr13,"\n");
  fprintf(fptr14,"\n");
  fprintf(fptr15,"\n");
  fprintf(fptr16,"\n");


  fclose(fptr1);
  fclose(fptr2);
  fclose(fptr3);
  fclose(fptr4);
  fclose(fptr5);
  fclose(fptr6);
  fclose(fptr7);
  fclose(fptr12);
  fclose(fptr13);
  fclose(fptr14);
  fclose(fptr15);
  fclose(fptr16);
}

void Transwrite (double *co, char s[10], int t)
{
  FILE *fptr1;
  int i;
  char datei[10];

  strcpy(datei,s);
  fptr1=fopen(datei,"a");
  //printf("OUT/%s",datei); // AN: To use this option, the strcpy(v[S].name,"OUT/S.csv") need to change into strcpy(v[S].name,"S.csv")
  //fptr1=fopen(("OUT/%s",datei),"a"); //modif MR
  fprintf(fptr1,"%d,",t); // , will be used as tab in CSV format, the first column will be the time saving step. d is integer, suitable for time step

  //Adding values for next columns (=next cell of C-GEM) for each time step
  for(i=1; i<=M; i++)
  {
    fprintf(fptr1,"%.10f,",co[i]); //f is float data with 10 digit after comma
  }
  fprintf(fptr1,"\n"); // enter for next row for next time step
  fclose(fptr1);
}

void Fluxwrite (int s, int t)
{
  FILE *fptr1;
  FILE *fptr2;
  char filename1[64];
  char filename2[64];
  int i;

  sprintf(filename1, "OUT/Flux/Advection_%s", v[s].name+4); //v[s].name is OUT/parameter.csv, so I need to remove 4 first characters (OUT/)
  sprintf(filename2, "OUT/Flux/Dispersion_%s", v[s].name+4);

  fptr1=fopen(filename1,"a");
  fptr2=fopen(filename2,"a");

  fprintf(fptr1,"%d,",t);
  fprintf(fptr2,"%d,",t);

  for(i=1; i<=M; i++)
  {
    fprintf(fptr1,"%f,",v[s].advflux[i]);// s is position of variables in chem list {Phy1, Phy2, Si, NO3, NH4, PO4, O2, TOC, S, SPM, DIC, AT, HS, PH, AlkC, CO2}
    fprintf(fptr2,"%f,",v[s].disflux[i]);
  }
  fprintf(fptr1,"\n");
  fprintf(fptr2,"\n");
  fclose(fptr1);
  fclose(fptr2);

}

void Rates (double *co, char s[50], int t)
{
  FILE *fptr1;
  int i;
  char datei[50];

  strcpy(datei,s);
  fptr1=fopen(datei,"a");
  //fptr1=fopen(("OUT/%s",datei),"a"); //modif MR
  fprintf(fptr1,"%d,",t);

  for(i=1; i<=M; i++)
  {
    fprintf(fptr1,"%.10f,",co[i]);
  }
  fprintf(fptr1,"\n");
  fclose(fptr1);
}


// read 2 column data from external files
void readFile (int datamax, double* gg, double* ff, char s[100])
{
	float data1out, data2out;
	FILE *fptr1;
	int i;
	char datei[100];

	strcpy(datei,s);
	printf("%s\n", datei);
	fptr1=fopen(datei,"r+"); //open file
	//fptr1=fopen(("OUT/%s",datei),"r+"); // modif MR

	if (fptr1==NULL)
        perror ("Error opening file");

	for ( i=0; i <datamax; i++)
	{
		fscanf(fptr1,"%f,%f\n",&data1out, &data2out); //f is float; "," is the space in CSV file. (sometimes it is \t-tab);"\n" is new line
		gg[i]=data1out;
		ff[i]=data2out;
	}
	fclose(fptr1);
}

