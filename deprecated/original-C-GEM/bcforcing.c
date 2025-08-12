/*____________________________________*/
/*bcforcing.c 			              */
/*tidal elevation at the mouth        */
/*river discharge Q                   */
/*last modified: 24/01/08 sa          */
/*____________________________________*/

#include "define.h"
#include "variables.h"

double Tide(int t)
{
    double resthour, hour;
    double omega, pfun, v0;

	int nhour;
	// Hourly values for the downstream boundary condition.
	hour=floor((double)t/(60.0*60.0))-((double)WARMUP)/(60.0*60.0);
	resthour=((double)t/(60.0*60.0))-((double)WARMUP)/(60.0*60.0)-hour;
	nhour=(int)hour;

	if (hour<0.0)
    {
        omega=2.*PI/3600.0;
   	    pfun= 0.080536912751677847;
	    v0=0.5*AMPL*sin(pfun*omega*t);
    }
        else {
	    v0=Hdata[nhour]+(Hdata[nhour+1]-Hdata[nhour])*resthour;
    }

        return v0;
}


double Discharge_ups(int t) //Freshwater inflow (upstream), This is the net discharge, NOT instant discharge!
{
	double Q_ups, restday, day;
	int nday;

	day=floor((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0);
	restday=((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0)-day;
	nday=(int)day;


	if(day<0.0) //Set Q to fixed value during warm-up. Please note that the input Q should be positive values
	{
		Q_ups=-Qdata[0];
	}else{
		Q_ups=-(Qdata[nday]+(Qdata[nday+1]-Qdata[nday])*restday); //linear interpolation of data
	}

	return Q_ups;
}

double Discharge_TT(int t) //Thi Tinh River discharge, cell=
{
	double Q_TT, restday, day;
	int nday;

	day=floor((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0);
	restday=((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0)-day;
	nday=(int)day;

	if(day<0.0) //Set Q to fixed value during warm-up
	{
		Q_TT=-Q_data_TT[0];}
	else{
		Q_TT=-(Q_data_TT[nday]+(Q_data_TT[nday+1]-Q_data_TT[nday])*restday); //linear interpolation of data
	}

	return Q_TT;
}

double Discharge_VT(int t) //Vam Thuat River, Cell=
{
    double Q_VT, restday, day;
    int nday;

    day=floor((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0);
    restday=((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0)-day;
    nday=(int)day;

    if(day<0.0) //Set Q to fixed value during warm-up
    {
        Q_VT=-Q_data_VT[0];
    }
    else
    {
        Q_VT=-(Q_data_VT[nday]+(Q_data_VT[nday+1]-Q_data_VT[nday])*restday); //linear interpolation of data
    }
    return Q_VT;
}


double Discharge_CANAL(int t) //Canals discharge, cell=
{
    double Q_CANAL, restday, day;
    int nday;

    day=floor((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0);
    restday=((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0)-day;
    nday=(int)day;

    if(day<0.0) //Set Q to fixed value during warm-up
    {
        Q_CANAL=-Q_data_CANAL[0];
    }
    else
    {
        Q_CANAL=-(Q_data_CANAL[nday]+(Q_data_CANAL[nday+1]-Q_data_CANAL[nday])*restday); //linear interpolation of data
    }
    return Q_CANAL;
}

double Discharge_DN(int t) //Dongnai River discharge, cell=
{
    double Q_DN, restday, day;
    int nday;

    day=floor((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0);
    restday=((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0)-day;
    nday=(int)day;

    if(day<0.0) //Set Q to fixed value during warm-up
    {
        Q_DN=-Q_data_DN[0];
    }
    else
    {
        Q_DN=-(Q_data_DN[nday]+(Q_data_DN[nday+1]-Q_data_DN[nday])*restday); //linear interpolation of data
    }
    return Q_DN;
}


double Discharge(int t, int i) //Total discharge
{
    double Q, Q_ups, Q_TT, Q_DN;
    double Q_VT, Q_CANAL;

    Q_DN=Discharge_DN(t); //DN, Dongnai River
    Q_CANAL=Discharge_CANAL(t); // Canals, including Nhieu Loc, Ben Nghe, Kenh Te canals
    Q_VT=Discharge_VT(t); //VT, Vam thuat River = Tham Luong canal
    Q_TT=Discharge_TT(t); //TT, Thi Tinh River
    Q_ups=Discharge_ups(t); // Upstream

    if (include_trib==1 && i<=31){
        Q=Q_ups+Q_TT+Q_VT+Q_CANAL+Q_DN;
    }else if (include_trib==1 && i<=37){
        Q=Q_ups+Q_TT+Q_VT+Q_CANAL;
    }else if (include_trib==1 && i<=45){
        Q=Q_ups+Q_TT+Q_VT;
    }else if (include_trib==1 && i<=61){
        Q=Q_ups+Q_TT;
    }else{
        Q=Q_ups;
    }
    return Q;
}

//Wind
double WS(int t)
{
	double WS, restday, day;
	int nday;

	day=floor((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0);
	restday=((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0)-day;
	nday=(int)day;

	if(day<0.0) //Set Wind speed to fixed value during warm-up
	{
		WS=winddata[0];
	}
	else
	{
		WS=(winddata[nday]+(winddata[nday+1]-winddata[nday])*restday); //linear interpolation of data
	}

	return WS;
}



double windspeed (int t, int i)
{
    double WindSpeed, Wspeed;
    Wspeed=WS(t);
    WindSpeed= Wspeed*(exp(-(((i+1)-1)*((double)DELXI))/((double)EL)));
    return WindSpeed;
}

void bgboundary(int t) //Boundary condition mainstream and tributaries
{
	double restday, day;
	int nday;

	day=floor((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0);
	restday=((double)t/(24.0*60.0*60.0))-((double)WARMUP)/(24.0*60.0*60.0)-day;
	nday=(int)day;

	if(day<0.0) //xw:add Sal
	{
		v[S].clb = sdatalb[0];
	}
	else
	{
	    v[S].clb=sdatalb[nday]+(sdatalb[nday+1]-sdatalb[nday])*restday;
	    //v[S].clb=sdatalb[nhour]+(sdatalb[nhour+1]-sdatalb[nhour])*resthour;
	}

	if(day<0.0)
	{
		v[TOC].clb = tocdatalb[0];
        v[TOC].cub = tocdataub[0];
        TT[TOC].c = toc_data_TT[0];
        DN[TOC].c = toc_data_DN[0];
        Canal[TOC].c = toc_data_CANAL[0];
        VT[TOC].c = toc_data_VT[0];
	}
	else
	{
        v[TOC].clb=tocdatalb[nday]+(tocdatalb[nday+1]-tocdatalb[nday])*restday;
	    v[TOC].cub=tocdataub[nday]+(tocdataub[nday+1]-tocdataub[nday])*restday;
	    TT[TOC].c=toc_data_TT[nday]+(toc_data_TT[nday+1]-toc_data_TT[nday])*restday;
	    DN[TOC].c=toc_data_DN[nday]+(toc_data_DN[nday+1]-toc_data_DN[nday])*restday;
	    Canal[TOC].c=toc_data_CANAL[nday]+(toc_data_CANAL[nday+1]-toc_data_CANAL[nday])*restday;
	    VT[TOC].c=toc_data_VT[nday]+(toc_data_VT[nday+1]-toc_data_VT[nday])*restday;
	}

    if(day<0.0)
	{
		v[O2].clb =O2datalb[0];
        v[O2].cub =O2dataub[0];
        TT[O2].c = O2_data_TT[0];
        DN[O2].c = O2_data_DN[0];
        Canal[O2].c = O2_data_CANAL[0];
        VT[O2].c = O2_data_VT[0];
	}
	else
	{
        v[O2].clb=O2datalb[nday]+(O2datalb[nday+1]-O2datalb[nday])*restday;
	    v[O2].cub=O2dataub[nday]+(O2dataub[nday+1]-O2dataub[nday])*restday;
	    TT[O2].c=O2_data_TT[nday]+(O2_data_TT[nday+1]-O2_data_TT[nday])*restday;
	    DN[O2].c=O2_data_DN[nday]+(O2_data_DN[nday+1]-O2_data_DN[nday])*restday;
	    Canal[O2].c=O2_data_CANAL[nday]+(O2_data_CANAL[nday+1]-O2_data_CANAL[nday])*restday;
	    VT[O2].c=O2_data_VT[nday]+(O2_data_VT[nday+1]-O2_data_VT[nday])*restday;
	}
    if(day<0.0)
	{

		v[NH4].clb =NH4datalb[0];
        v[NH4].cub =NH4dataub[0];
        TT[NH4].c = NH4_data_TT[0];
        DN[NH4].c = NH4_data_DN[0];
        Canal[NH4].c = NH4_data_CANAL[0];
        VT[NH4].c = NH4_data_VT[0];
	}
	else
	{
        v[NH4].clb=NH4datalb[nday]+(NH4datalb[nday+1]-NH4datalb[nday])*restday;
	    v[NH4].cub=NH4dataub[nday]+(NH4dataub[nday+1]-NH4dataub[nday])*restday;
	    TT[NH4].c=NH4_data_TT[nday]+(NH4_data_TT[nday+1]-NH4_data_TT[nday])*restday;
	    DN[NH4].c=NH4_data_DN[nday]+(NH4_data_DN[nday+1]-NH4_data_DN[nday])*restday;
	    Canal[NH4].c=NH4_data_CANAL[nday]+(NH4_data_CANAL[nday+1]-NH4_data_CANAL[nday])*restday;
	    VT[NH4].c=NH4_data_VT[nday]+(NH4_data_VT[nday+1]-NH4_data_VT[nday])*restday;
	}
    if(day<0.0)
	{
		v[NO3].clb =NO3datalb[0];
        v[NO3].cub = NO3dataub[0];
        TT[NO3].c = NO3_data_TT[0];
        DN[NO3].c = NO3_data_DN[0];
        Canal[NO3].c = NO3_data_CANAL[0];
        VT[NO3].c = NO3_data_VT[0];
	}
	else
	{
        v[NO3].clb=NO3datalb[nday]+(NO3datalb[nday+1]-NO3datalb[nday])*restday;
	    v[NO3].cub=NO3dataub[nday]+(NO3dataub[nday+1]-NO3dataub[nday])*restday;
	    TT[NO3].c=NO3_data_TT[nday]+(NO3_data_TT[nday+1]-NO3_data_TT[nday])*restday;
	    DN[NO3].c=NO3_data_DN[nday]+(NO3_data_DN[nday+1]-NO3_data_DN[nday])*restday;
	    Canal[NO3].c=NO3_data_CANAL[nday]+(NO3_data_CANAL[nday+1]-NO3_data_CANAL[nday])*restday;
	    VT[NO3].c=NO3_data_VT[nday]+(NO3_data_VT[nday+1]-NO3_data_VT[nday])*restday;
	}
    if(day<0.0)
	{
		v[Phy1].clb = phy1datalb[0];
        v[Phy1].cub = phy1dataub[0];
        TT[Phy1].c = phy1_data_TT[0];
        DN[Phy1].c = phy1_data_DN[0];
        Canal[Phy1].c = phy1_data_CANAL[0];
        VT[Phy1].c = phy1_data_VT[0];
	}
	else
	{
        v[Phy1].clb=phy1datalb[nday]+(phy1datalb[nday+1]-phy1datalb[nday])*restday;
	    v[Phy1].cub = phy1dataub[nday]+(phy1dataub[nday+1]-phy1dataub[nday])*restday;
	    TT[Phy1].c=phy1_data_TT[nday]+(phy1_data_TT[nday+1]-phy1_data_TT[nday])*restday;
	    DN[Phy1].c=phy1_data_DN[nday]+(phy1_data_DN[nday+1]-phy1_data_DN[nday])*restday;
	    Canal[Phy1].c=phy1_data_CANAL[nday]+(phy1_data_CANAL[nday+1]-phy1_data_CANAL[nday])*restday;
	    VT[Phy1].c=phy1_data_VT[nday]+(phy1_data_VT[nday+1]-phy1_data_VT[nday])*restday;
	}
    if(day<0.0)
    {
        v[Phy2].clb = phy2datalb[0];
        v[Phy2].cub = phy2dataub[0];
        TT[Phy2].c = phy2_data_TT[0];
        DN[Phy2].c = phy2_data_DN[0];
        Canal[Phy2].c = phy2_data_CANAL[0];
        VT[Phy2].c = phy2_data_VT[0];
    }
    else
    {
        v[Phy2].clb=phy2datalb[nday]+(phy2datalb[nday+1]-phy2datalb[nday])*restday;
        v[Phy2].cub= (phy2dataub[nday]+(phy2dataub[nday+1]-phy2dataub[nday])*restday);
	    TT[Phy2].c=phy2_data_TT[nday]+(phy2_data_TT[nday+1]-phy2_data_TT[nday])*restday;
	    DN[Phy2].c=phy2_data_DN[nday]+(phy2_data_DN[nday+1]-phy2_data_DN[nday])*restday;
	    Canal[Phy2].c=phy2_data_CANAL[nday]+(phy2_data_CANAL[nday+1]-phy2_data_CANAL[nday])*restday;
	    VT[Phy2].c=phy2_data_VT[nday]+(phy2_data_VT[nday+1]-phy2_data_VT[nday])*restday;
    }
    if(day<0.0)
	{
		v[Si].clb = sidatalb[0];
        v[Si].cub = sidataub[0];
        TT[Si].c = si_data_TT[0];
        DN[Si].c = si_data_DN[0];
        Canal[Si].c = si_data_CANAL[0];
        VT[Si].c = si_data_VT[0];
	}
	else
	{
        v[Si].clb=sidatalb[nday]+(sidatalb[nday+1]-sidatalb[nday])*restday;
	    v[Si].cub=sidataub[nday]+(sidataub[nday+1]-sidataub[nday])*restday;
	    TT[Si].c=si_data_TT[nday]+(si_data_TT[nday+1]-si_data_TT[nday])*restday;
	    DN[Si].c=si_data_DN[nday]+(si_data_DN[nday+1]-si_data_DN[nday])*restday;
	    Canal[Si].c=si_data_CANAL[nday]+(si_data_CANAL[nday+1]-si_data_CANAL[nday])*restday;
	    VT[Si].c=si_data_VT[nday]+(si_data_VT[nday+1]-si_data_VT[nday])*restday;
	}
    if(day<0.0)
    {
        v[PO4].clb =po4datalb[0];
        v[PO4].cub =po4dataub[0];
        TT[PO4].c = po4_data_TT[0];
        DN[PO4].c = po4_data_DN[0];
        Canal[PO4].c = po4_data_CANAL[0];
        VT[PO4].c = po4_data_VT[0];
    }
    else
    {
        v[PO4].clb=po4datalb[nday]+(po4datalb[nday+1]-po4datalb[nday])*restday;
        v[PO4].cub=po4dataub[nday]+(po4dataub[nday+1]-po4dataub[nday])*restday;
	    TT[PO4].c=po4_data_TT[nday]+(po4_data_TT[nday+1]-po4_data_TT[nday])*restday;
	    DN[PO4].c=po4_data_DN[nday]+(po4_data_DN[nday+1]-po4_data_DN[nday])*restday;
	    Canal[PO4].c=po4_data_CANAL[nday]+(po4_data_CANAL[nday+1]-po4_data_CANAL[nday])*restday;
	    VT[PO4].c=po4_data_VT[nday]+(po4_data_VT[nday+1]-po4_data_VT[nday])*restday;
    }
    if(day<0.0)
    {
        v[SPM].clb =spmdatalb[0];
        v[SPM].cub =spmdataub[0];
        TT[SPM].c = spm_data_TT[0];
        DN[SPM].c = spm_data_DN[0];
        Canal[SPM].c = spm_data_CANAL[0];
        VT[SPM].c = spm_data_VT[0];
    }
    else
    {
        v[SPM].clb=spmdatalb[nday]+(spmdatalb[nday+1]-spmdatalb[nday])*restday;
        v[SPM].cub=spmdataub[nday]+(spmdataub[nday+1]-spmdataub[nday])*restday;
	    TT[SPM].c=spm_data_TT[nday]+(spm_data_TT[nday+1]-spm_data_TT[nday])*restday;
	    DN[SPM].c=spm_data_DN[nday]+(spm_data_DN[nday+1]-spm_data_DN[nday])*restday;
	    Canal[SPM].c=spm_data_CANAL[nday]+(spm_data_CANAL[nday+1]-spm_data_CANAL[nday])*restday;
	    VT[SPM].c=spm_data_VT[nday]+(spm_data_VT[nday+1]-spm_data_VT[nday])*restday;
    }
}
