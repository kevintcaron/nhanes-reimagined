/********
File:		Conf_Int_Exp.sas

Purpose:	Replicate Confidence Intervals in Exposure Report
                for 2013-2014 Total Blood Mercury 95th percentile
                estimates for 6-11, 12-19, and 20+ age groups,
                males and females, and various racial groups;

Date:		22 APR 05

Date Revised:	26 APR 05
		05 MAY 05
                27 Oct 08
                23 Jun 11
                02 Feb 17

Input Datasets:	PbCd_h and demo_h

Programmer:	Lisa Mirel / Sam Caudill
*************************************************/
options nomprint nosymbolgen nomlogic nonotes nosource nosource2;

***bring in datasets;

LIBNAME l06  XPORT 'C:\Users\wrn0\GitHub\ntcp-data-explorer\PBCD_H.XPT';
Libname demo xport 'C:\Users\wrn0\GitHub\ntcp-data-explorer\DEMO_H.XPT';

***Note: default method for CI of percentile in SUDAAN uses the Logit;

data lab6;
	set  l06.PbCd_h; 
	run;
data demo;
	set  demo.demo_h; 
	run;

proc sort data=lab6;
	by seqn;
proc sort data=demo;
	by seqn;
data l6dem;
	merge lab6 (in=a) demo (in=b);
	by seqn;
	if a=1;

*Define Age Groups;
 
  if ridageyr ge  6 and ridageyr le 11 then age_grp = 1;
  if ridageyr ge 12 and ridageyr le 19 then age_grp = 2;
  if ridageyr ge 20 then age_grp = 3;
  if ridageyr ge  6 then age_group = 1;

  if riagendr eq 1 then gender = 1;
  if riagendr eq 2 then gender = 2;

  if ridreth3 eq 1 then race = 1; *'MA';
  *if ridreth3 eq 2 then race = 2; *'OH'; *An estimate for this category by itself is not calculated.  Instead OH is combined with MA to create AH; 
  if ridreth3 eq 3 then race = 2; *'NHW';
  if ridreth3 eq 4 then race = 3; *'NHB';
  if ridreth3 eq 6 then race = 4; *'NHA';
  if ridreth3 eq 7 then race = 5; *Non-Hispanic Multi-racial;
  racial = 2;
  if ridreth3 eq 1 or ridreth3 eq 2 then racial = 1; *'AH';

	run; 

******

	Macro for calculating totals -- following steps in 
	Appendix A Fourth Exposure Report:
	Confidence Interval Estimation for Percentiles

******;
%macro pcntci (var1, var2, var3, var4, var5, var6, var7);
data L6dem2;
	set L6dem;
	mvar=1;
    anal_var_orig = &var6;
    wt_orig = &var7;
    wt_orig_rnd = round(wt_orig,1); 
	run;

***Step 1a;
proc univariate data = l6dem2;
	where &var1=&var2 and sddsrvyr=&var3;
	var anal_var_orig;
   	output out=xpercenta pctlpre=P_ pctlpts=&var4 n=wtn; 
   	freq wt_orig_rnd;
run;

data xpercent1a;
	set xpercenta;
	mvar=1;
	RUN;
proc sort data=l6dem2;
	by mvar;
proc sort data=xpercent1a;
	by mvar;
data xchperc1a;
	merge l6dem2 (in=a) xpercent1a (in=b);
	by mvar;
	run;

***Step 1b;
  proc sort data=l6dem2;
 by anal_var_orig;

proc means noprint data=l6dem2;
 by anal_var_orig;
 where &var1=&var2 and sddsrvyr=&var3;
var wt_orig;
output out=xwt_mean mean=wt_mean; *wt_mean is now the mean weight of all
                                   subjects in the same domain/subsample
                                   with the same measured result;
run;

data xl6dem2;
 merge xwt_mean l6dem2;
  by anal_var_orig;
    wt_mean_rnd = round(wt_mean,1); 
run;

data xxl6dem2;
 set xl6dem2;
  by anal_var_orig;

 if first.anal_var_orig then do;
  num = 1;
 end;
 else do;
  num + 1;
 end;
   anal_var_incr = anal_var_orig + num/1000000000;  
  run;

proc univariate data = xxl6dem2;
	where &var1=&var2 and sddsrvyr=&var3;
	var anal_var_incr;
   	output out=xpercentb pctlpre=P_ pctlpts=&var4 n=wtn; 
   	freq wt_mean_rnd;
run;

data xpercent1b;
	set xpercentb;
	mvar=1;
	RUN;
proc sort data=xxl6dem2;
	by mvar;
proc sort data=xpercent1b;
	by mvar;
data xchperc1b;
	merge xxl6dem2 (in=a) xpercent1b (in=b);
	by mvar;
	run;

***Step 2a;
data xchperc2a;
	set xchperc1a;
	if &var1=&var2 and sddsrvyr=&var3 and 0<=anal_var_orig<P_&var4 then ind2=1;
	else if &var1=&var2 and sddsrvyr=&var3 and anal_var_orig>=P_&var4 then ind2=0;
	run;

proc sort data=xchperc2a;
	by sdmvstra sdmvpsu;
proc descript data=xchperc2a design=wr atlevel1=1 atlevel2=2 ; 
SUBPOPN &var1=&var2 and sddsrvyr=&var3;
nest sdmvstra sdmvpsu /missunit;
  weight wt_orig;                  *BE SURE TO USE THE PROPER WEIGHT; 
  class ind2;
  var ind2;  
print 	nsum="samsize" mean semean geomean segeomean
  		deffmean="deff #4"/style=nchs nsumfmt=f8.0 meanfmt=f9.4 semeanfmt=f9.4
		geomeanfmt=f5.3  segeomeanfmt=f5.3;
output 	atlev1 atlev2 nsum mean semean geomean segeomean deffmean/filename=xest1 replace
		nsumfmt=f8.0 meanfmt=f9.4 semeanfmt=f9.4
		geomeanfmt=f5.3  segeomeanfmt=f5.3;
run;
DATA xpest2a;
  SET xest1;
  if _N_=1;
  mvar = 1;
  semean_orig = semean;
deffmean = max(1,deffmean);
  deffmean_orig = deffmean;
  drop nsum mean semean geomean segeomean deffmean atlev2 atlev1;
run;
proc sort data=xpest2a;
	by mvar;
proc sort data=xPERCENT1a;
	by mvar;
run;


***Step 2b;
data xchperc2b;
	set xchperc1b;
	if &var1=&var2 and sddsrvyr=&var3 and 0<=anal_var_incr<P_&var4 then ind2=1;
	else if &var1=&var2 and sddsrvyr=&var3 and anal_var_incr>=P_&var4 then ind2=0;
	run;

proc sort data=xchperc2b;
	by sdmvstra sdmvpsu;
proc descript data=xchperc2b design=wr atlevel1=1 atlevel2=2 ; 
SUBPOPN &var1=&var2 and sddsrvyr=&var3;
nest sdmvstra sdmvpsu /missunit;
  weight wt_mean;                  *BE SURE TO USE THE PROPER WEIGHT; 
  class ind2;
  var ind2;  
print 	nsum="samsize" mean semean geomean segeomean
  		deffmean="deff #4"/style=nchs nsumfmt=f8.0 meanfmt=f9.4 semeanfmt=f9.4
		geomeanfmt=f5.3  segeomeanfmt=f5.3;
output 	atlev1 atlev2 nsum mean semean geomean segeomean deffmean/filename=xest2 replace
		nsumfmt=f8.0 meanfmt=f9.4 semeanfmt=f9.4
		geomeanfmt=f5.3  segeomeanfmt=f5.3;
run;
DATA xpest2b;
  SET xest2;
  if _N_=1;
  mvar = 1;
  ddf=atlev2-atlev1;
run;
proc sort data=xpest2b;
	by mvar;
proc sort data=xPERCENT1b;
	by mvar;
run;

***Step 3;

******************************************************************;
*The forumlas of Korn et al are used to estimate the proportion of
 subjects below the selected percentile-- from Sam Caudill code
******************************************************************;
data xtest;
  merge xpest2a xpest2b xPERCENT1b; 
  by mvar;
  N_ACT =NSUM;                *ACTUAL SAMPLE SIZE; 
  PT = MEAN;                  *SUDAAN WEIGHTED MEAN PROPORTION;
  CALL SYMPUT('N_ACT',NSUM);
  T_NUM = TINV(0.975,NSUM-1);
  T_DEN = TINV(0.975,ddf); 
  N1 = ((T_NUM/T_DEN)**2)*N_ACT/DEFFMEAN_orig; *EFFECTIVE SAMPLE SIZE - SAM METHOD;
  N=((T_NUM/T_DEN)**2)*MEAN*(1 - MEAN)/(SEMEAN_orig**2); *EFFECTIVE SAMPLE SIZE DUE TO; 
                                       *COMPLEX STRATIFIED SAMPLING - KORN METHOD;
  IF N eq . then N = N1;
  IF N GT NSUM THEN N = NSUM;
  IF MEAN EQ 0.0 THEN N = NSUM; 
  NA=MEAN*N;                    *EFFECTIVE NUMBER OF SUBJECTS;
  OUTPUT;
RUN;

DATA xCYTO; 
  SET xtest;
  V1 = 2*NA;
  V2 = 2*(N - NA + 1);
  V3 = 2*(NA + 1);
  V4 = 2*(N - NA);
  PL = V1*FINV(0.025,V1,V2)/(V2 + V1*FINV(0.025,V1,V2));
  PU = V3*FINV(0.975,V3,V4)/(V4 + V3*FINV(0.975,V3,V4));
  PT = PT*100;
  N_EFF = N;
RUN;

DATA xxCYTO; 
  SET xCYTO;
L95 = PL*100;  
U95 = PU*100;  

IF L95 EQ . THEN L95 = 0.0;
IF U95 EQ . THEN U95 = 100.0;

  %IF &var4 EQ 10 %THEN %DO; PT = 10.0 ;  %END; 
  %IF &var4 EQ 25 %THEN %DO; PT = 25.0 ;  %END; 
  %IF &var4 EQ 50 %THEN %DO; PT = 50.0 ;  %END; 
  %IF &var4 EQ 75 %THEN %DO; PT = 75.0 ;  %END; 
  %IF &var4 EQ 90 %THEN %DO; PT = 90.0 ;  %END; 
  %IF &var4 EQ 95 %THEN %DO; PT = 95.0 ;  %END; 

   if L95 gt PT then L95 = PT;
   if U95 lt PT then U95 = PT;

  title3 "PERCENTILE (WITH CIs)";
RUN;

***Step 4;
data xXtest;
	set xxcyto;
	CALL SYMPUT('L95',LEFT(PUT(L95,8.1)));
  	CALL SYMPUT('U95',LEFT(PUT(U95,8.1)));
  	CALL SYMPUT('MEAN',LEFT(PUT(PT,8.1)));
	mvar=1;
	run;
proc sort data=XXtest;
	by mvar;
proc sort data=xxl6dem2;
	by mvar;
data xcomp;
	merge XXtest (in=a) xxl6dem2 (in=b);
	by mvar;
	if b=1;
	run;

proc univariate data = xcomp;
	var anal_var_incr;
	where &var1=&var2 and sddsrvyr=&var3;
   	output out=xxEST pctlpre=P_ pctlpts=&L95 &U95 &MEAN 
	PCTLPRE = A 
  	PCTLNAME = L95 U95 MEAN ; 
  	freq wt_mean_rnd;
   run;
proc print data=xxEST;
run;
*creates multiple final dataset to be put in a final large dataset with all
datapoints;
data fin&var5;
	set xxest;
	
	if "&var1" eq 'age_grp' and &var2 eq 1 then Population_Group = "Age:  6-11";
	if "&var1" eq 'age_grp' and &var2 eq 2 then Population_Group = "Age: 12-19";
	if "&var1" eq 'age_grp' and &var2 eq 3 then Population_Group = "Age: 20+  ";
	if "&var1" eq 'age_group' and &var2 eq 1 then Population_Group = "Age: ALL";
	if "&var1" eq 'gender' and &var2 eq 1 then Population_Group = "MALE  ";
	if "&var1" eq 'gender' and &var2 eq 2 then Population_Group = "FEMALE";
	if "&var1" eq 'race' and &var2 eq 1 then Population_Group = "MA    ";
	if "&var1" eq 'race' and &var2 eq 2 then Population_Group = "NHW   ";
	if "&var1" eq 'race' and &var2 eq 3 then Population_Group = "NHB   ";
 	if "&var1" eq 'race' and &var2 eq 4 then Population_Group = "NHA   ";
	if "&var1" eq 'racial' and &var2 eq 1 then Population_Group = "AH    ";


    yr=&var3;
	per=&var4;
	anly="&var6";
	P_MEAN = round(P_MEAN,0.01);
	P_L95 = round(P_L95,0.01);
	P_U95 = round(P_U95,0.01);
	perci=compress (P_MEAN||"("||P_L95||"-"||P_U95||")");
		N_ACT = &N_ACT;
	keep anly Population_Group yr per perci 	N_ACT;
run;
proc print data=fin&var5 noobs;
var  Population_Group yr per perci N_ACT;
run;
%mend;

**			Variable definitions used in Macro

VAR1 = SubPopulation Variable Name
VAR2 = SubPopulation Value to Select
VAR3 = SURVEY YEAR
VAR4 = PERCENTILE
VAR5 = INDICATES DATASET NUMBER for Concatination of All Data Sets
VAR6 = ANALYTE OF INTEREST
VAR7 = WEIGHT VARIABLE;

***NOTE: CAN ADD MORE MACRO VARIABLES IN CODE TO SELECT ON GENDER AND RACE/ETHNICTY;

/* Testing with Wanzhe
proc surveymeans data=l6dem median q3 nonsymcl;
	strata sdmvstra;
	cluster sdmvpsu;
	weight wtmec2yr;
	domain age_grp;
	var lbxthg;
	run;
*/

%pcntci (age_grp, 1, 8, 95, 1, lbxthg, wtsh2yr);
%pcntci (age_grp, 2, 8, 95, 2, lbxthg, wtsh2yr);
%pcntci (age_grp, 3, 8, 95, 3, lbxthg, wtsh2yr);
%pcntci (age_group, 1, 8, 95, 4, lbxthg, wtsh2yr);

%pcntci (gender, 1, 8, 95, 5, lbxthg, wtmec2yr);
%pcntci (gender, 2, 8, 95, 6, lbxthg, wtmec2yr);

%pcntci (race, 1, 8, 95, 7, lbxthg, wtmec2yr);
%pcntci (race, 2, 8, 95, 8, lbxthg, wtmec2yr);
%pcntci (race, 3, 8, 95, 9, lbxthg, wtmec2yr);
%pcntci (race, 4, 8, 95, 10, lbxthg, wtmec2yr);
%pcntci (racial, 1, 8, 95, 11, lbxthg, wtmec2yr);

***CREATE ONE LARGE DATASET USING ALL DATASETS CREATED IN THE MACRO;
options nocenter;
data allperc;
	set fin1-fin11; 
	run;
proc print data=allperc;
run;
Quit;
