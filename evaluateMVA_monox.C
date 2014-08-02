/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Exectuable: TMVAClassificationApplication                                      *
 *                                                                                *
 * This macro provides a simple example on how to use the trained classifiers     *
 * within an analysis module                                                      *
 **********************************************************************************/

#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>
#include <string>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"
#include "TChain.h"
#include "TChainElement.h"
#include "TBranch.h"
#include "TRandom.h"

#include "Math/LorentzVector.h"
#include "Math/VectorUtil.h"
#include "TMath.h"

#include "MitMonoJet/macros/TMVAGui.C"

#include "MitMonoJet/Core/MitDMSTree.h"
#include "MitMonoJet/macros/functions.C"

#if not defined(__CINT__) || defined(__MAKECINT__)
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"
#endif

using namespace std;
using namespace TMVA;

//--------------------------------------------------------------------

void evaluateMVA_monox(
TString inputFile    = "/afs/cern.ch/work/c/ceballos/mva/samples/ttbar.root", 
TString myMethodList = "BDTG,MLP,LikelihoodD",
TString myVarList    = "QGtag,tau1,tau2,tau2tau1,jetC2b0,jetC2b0p2,jetC2b0p5,jetC2b1,jetC2b2,jetQJetVol,jetMassSDb0,jetMassSDb1,jetMassSDb2,jetMassSDbm1,jetMassPruned,jetMassFiltered,jetMassTrimmed,jetMassRaw,jetPull,jetPullAngle,jetQGtagSub1,jetQGtagSub2,jetQGtagComb",
TString outTag       = "default",
TString pathWeights  = "weights/",
TString suffix       = "ww",
Int_t   nsel         = 0,
Int_t   count        = 0
) {   
#ifdef __CINT__
  gROOT->ProcessLine( ".O0" ); // turn off optimization in CINT
#endif

  cout << "Looking for weights dir at " << pathWeights << endl;
  
  vector<TString> samples;
  samples.push_back(inputFile.Data());

  //--------------------------------------------------------------------------------
  // IMPORTANT: set the following variables to the same set used for MVA training!!!
  //--------------------------------------------------------------------------------
  std::map<std::string,int> mvaVar;

  mvaVar["QGtag"]	    = 1;
  mvaVar["tau1"]            = 1;
  mvaVar["tau2"]            = 1;
  mvaVar["tau2tau1"]	    = 1;
  mvaVar["jetC2b0"]	    = 1;
  mvaVar["jetC2b0p2"]       = 1;
  mvaVar["jetC2b0p5"]       = 1;
  mvaVar["jetC2b1"]	    = 1;
  mvaVar["jetC2b2"]	    = 1;
  mvaVar["jetQJetVol"]      = 1;
  mvaVar["jetMassSDb0"]     = 1;
  mvaVar["jetMassSDb1"]     = 1;
  mvaVar["jetMassSDb2"]     = 1;
  mvaVar["jetMassSDbm1"]    = 1;
  mvaVar["jetMassPruned"]   = 1;
  mvaVar["jetMassFiltered"] = 1;
  mvaVar["jetMassTrimmed"]  = 1;
  mvaVar["jetMassRaw"]      = 1;
  mvaVar["jetPull"]         = 1;
  mvaVar["jetPullAngle"]    = 1;
  mvaVar["jetQGtagSub1"]    = 1;
  mvaVar["jetQGtagSub2"]    = 1;
  mvaVar["jetQGtagComb"]    = 1;
  mvaVar["frozen"]          = 1;

  //---------------------------------------------------------------
  // specifies the selection applied to events in the training
  //---------------------------------------------------------------

  // This loads the library
  TMVA::Tools::Instance();

  // Default MVA methods to be trained + tested
  std::map<std::string,int> Use;

  // --- Cut optimisation
  Use["Cuts"]            = 0;
  Use["CutsD"]           = 0;
  Use["CutsPCA"]         = 0;
  Use["CutsGA"]          = 0;
  Use["CutsSA"]          = 0;
  // 
  // --- 1-dimensional likelihood ("naive Bayes estimator")
  Use["Likelihood"]      = 0;
  Use["LikelihoodD"]     = 0; // the "D" extension indicates decorrelated input variables (see option strings)
  Use["LikelihoodPCA"]   = 0; // the "PCA" extension indicates PCA-transformed input variables (see option strings)
  Use["LikelihoodKDE"]   = 0;
  Use["LikelihoodMIX"]   = 0;
  //
  // --- Mutidimensional likelihood and Nearest-Neighbour methods
  Use["PDERS"]           = 0;
  Use["PDERSD"]          = 0;
  Use["PDERSPCA"]        = 0;
  Use["PDEFoam"]         = 0;
  Use["PDEFoamBoost"]    = 0; // uses generalised MVA method boosting
  Use["KNN"]             = 0; // k-nearest neighbour method
  //
  // --- Linear Discriminant Analysis
  Use["LD"]              = 0; // Linear Discriminant identical to Fisher
  Use["Fisher"]          = 0;
  Use["FisherG"]         = 0;
  Use["BoostedFisher"]   = 0; // uses generalised MVA method boosting
  Use["HMatrix"]         = 0;
  //
  // --- Function Discriminant analysis
  Use["FDA_GA"]          = 0; // minimisation of user-defined function using Genetics Algorithm
  Use["FDA_SA"]          = 0;
  Use["FDA_MC"]          = 0;
  Use["FDA_MT"]          = 0;
  Use["FDA_GAMT"]        = 0;
  Use["FDA_MCMT"]        = 0;
  //
  // --- Neural Networks (all are feed-forward Multilayer Perceptrons)
  Use["MLP"]             = 0; // Recommended ANN
  Use["MLPBFGS"]         = 0; // Recommended ANN with optional training method
  Use["MLPBNN"]          = 0; // Recommended ANN with BFGS training method and bayesian regulator
  Use["CFMlpANN"]        = 0; // Depreciated ANN from ALEPH
  Use["TMlpANN"]         = 0; // ROOT's own ANN
  //
  // --- Support Vector Machine 
  Use["SVM"]             = 0;
  // 
  // --- Boosted Decision Trees
  Use["BDT"]             = 0; // uses Adaptive Boost
  Use["BDTG"]            = 0; // uses Gradient Boost
  Use["BDTB"]            = 0; // uses Bagging
  Use["BDTD"]            = 0; // decorrelation + Adaptive Boost
  // 
  // --- Friedman's RuleFit method, ie, an optimised series of cuts ("rules")
  Use["RuleFit"]         = 0;
  // ---------------------------------------------------------------
  Use["Plugin"]          = 0;
  Use["Category"]        = 0;
  Use["SVM_Gauss"]       = 0;
  Use["SVM_Poly"]        = 0;
  Use["SVM_Lin"]         = 0;

  std::cout << std::endl;
  std::cout << "==> Start TMVAClassificationApplication" << std::endl;

  // Select methods (don't look at this code - not of interest)
  if (myMethodList != "") {
    for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;

    std::vector<TString> mlist = gTools().SplitString( myMethodList, ',' );
    for (UInt_t i=0; i<mlist.size(); i++) {
      std::string regMethod(mlist[i]);

      if (Use.find(regMethod) == Use.end()) {
        std::cout << "Method \"" << regMethod 
                  << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
        for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
          std::cout << it->first << " ";
        }
        std::cout << std::endl;
        return;
      }
      Use[regMethod] = 1;
    }
  }

  if (myVarList != "") {
    for (std::map<std::string,int>::iterator it = mvaVar.begin(); it != mvaVar.end(); it++) it->second = 0;

    std::vector<TString> mlist = TMVA::gTools().SplitString( myVarList, ',' );
    for (UInt_t i=0; i<mlist.size(); i++) {
      std::string regMethod(mlist[i]);

      if (mvaVar.find(regMethod) == mvaVar.end()) {
        std::cout << "Method \"" << regMethod << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
        for (std::map<std::string,int>::iterator it = mvaVar.begin(); it != mvaVar.end(); it++) std::cout << it->first << " ";
        std::cout << std::endl;
        return;
      }
      mvaVar[regMethod] = 1;
    }
  }

  // --------------------------------------------------------------------------------------------------

  const unsigned int nsamples = samples.size();
  
  TString *allEvts = new TString("hDAllEvents");
  TString *dirFwk  = new TString("AnaFwkMod");

  for( unsigned int i = 0 ; i < nsamples ; ++i ){

    // --- Create the Reader object

    TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );    

    // Create a set of variables and declare them to the reader
    // - the variable names MUST corresponds in name and type to those given in the weight file(s) used
    //    Float_t var1, var2;
    //    Float_t var3, var4;
    //    reader->AddVariable( "myvar1 := var1+var2", &var1 );
    //    reader->AddVariable( "myvar2 := var1-var2", &var2 );
    //    reader->AddVariable( "var3",                &var3 );
    //    reader->AddVariable( "var4",                &var4 );

    Float_t QGtag;	     
    Float_t tau1;
    Float_t tau2;
    Float_t tau2tau1;
    Float_t jetC2b0;
    Float_t jetC2b0p2;
    Float_t jetC2b0p5;
    Float_t jetC2b1;
    Float_t jetC2b2;
    Float_t jetQJetVol;
    Float_t jetMassSDb0;
    Float_t jetMassSDb1;
    Float_t jetMassSDb2;
    Float_t jetMassSDbm1;
    Float_t jetMassPruned;  
    Float_t jetMassFiltered;
    Float_t jetMassTrimmed; 
    Float_t jetMassRaw;
    Float_t jetPull;
    Float_t jetPullAngle;
    Float_t jetQGtagSub1;
    Float_t jetQGtagSub2;
    Float_t jetQGtagComb;
    Float_t frozen;

    if (mvaVar["QGtag"])	   reader->AddVariable( "QGtag",	   &QGtag	   );
    if (mvaVar["tau1"])            reader->AddVariable( "tau1",            &tau1           );
    if (mvaVar["tau2"])            reader->AddVariable( "tau2",            &tau2           );
    if (mvaVar["tau2tau1"])	   reader->AddVariable( "tau2tau1",	   &tau2tau1       );
    if (mvaVar["jetC2b0"])	   reader->AddVariable( "jetC2b0",	   &jetC2b0	   );
    if (mvaVar["jetC2b0p2"])	   reader->AddVariable( "jetC2b0p2",	   &jetC2b0p2      );
    if (mvaVar["jetC2b0p5"])	   reader->AddVariable( "jetC2b0p5",	   &jetC2b0p5      );
    if (mvaVar["jetC2b1"])	   reader->AddVariable( "jetC2b1",	   &jetC2b1	   );
    if (mvaVar["jetC2b2"])	   reader->AddVariable( "jetC2b2",	   &jetC2b2	   );
    if (mvaVar["jetQJetVol"])	   reader->AddVariable( "jetQJetVol",	   &jetQJetVol     );
    if (mvaVar["jetMassSDb0"])     reader->AddVariable( "jetMassSDb0",     &jetMassSDb0    );
    if (mvaVar["jetMassSDb1"])     reader->AddVariable( "jetMassSDb1",     &jetMassSDb1    );
    if (mvaVar["jetMassSDb2"])     reader->AddVariable( "jetMassSDb2",     &jetMassSDb2    );
    if (mvaVar["jetMassSDbm1"])    reader->AddVariable( "jetMassSDbm1",    &jetMassSDbm1   );
    if (mvaVar["jetMassPruned"])   reader->AddVariable( "jetMassPruned",   &jetMassPruned  );
    if (mvaVar["jetMassFiltered"]) reader->AddVariable( "jetMassFiltered", &jetMassFiltered);
    if (mvaVar["jetMassTrimmed"])  reader->AddVariable( "jetMassTrimmed",  &jetMassTrimmed );
    if (mvaVar["jetMassRaw"])	   reader->AddVariable( "jetMassRaw",	   &jetMassRaw     );
    if (mvaVar["jetPull"])         reader->AddVariable( "jetPull",         &jetPull        );
    if (mvaVar["jetPullAngle"])    reader->AddVariable( "jetPullAngle",    &jetPullAngle   );
    if (mvaVar["jetQGtagSub1"])    reader->AddVariable( "jetQGtagSub1",    &jetQGtagSub1   );
    if (mvaVar["jetQGtagSub2"])    reader->AddVariable( "jetQGtagSub2",    &jetQGtagSub2   );
    if (mvaVar["jetQGtagComb"])    reader->AddVariable( "jetQGtagComb",    &jetQGtagComb   );
    if (mvaVar["frozen"])          reader->AddVariable( "frozen",          &frozen         );

    // Spectator variables declared in the training have to be added to the reader, too
    //    Float_t spec1,spec2;
    //    reader->AddSpectator( "spec1 := var1*2",   &spec1 );
    //    reader->AddSpectator( "spec2 := var1*3",   &spec2 );

    //Float_t Category_cat1, Category_cat2, Category_cat3;
    if (Use["Category"]){
      // Add artificial spectators for distinguishing categories
      //       reader->AddSpectator( "Category_cat1 := var3<=0",             &Category_cat1 );
      //       reader->AddSpectator( "Category_cat2 := (var3>0)&&(var4<0)",  &Category_cat2 );
      //       reader->AddSpectator( "Category_cat3 := (var3>0)&&(var4>=0)", &Category_cat3 );
    }

    // --- Book the MVA methods

    TString prefix = outTag;

    // Book method(s)
    for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
      if (it->second) {
        TString methodName = TString(it->first) + TString(" method");
        TString weightfile = pathWeights + prefix + TString("_") + TString(it->first) + TString(".weights.xml");
        reader->BookMVA( methodName, weightfile ); 
      }
    }

    // Prepare input tree (this must be replaced by your data source)
    // in this example, there is a toy tree with signal and one with background events
    // we'll later on use only the "signal" events for the test in this example.
    //   

    TFile fif(samples.at(i).Data());
    if(fif.GetDirectory(dirFwk->Data())) fif.cd(dirFwk->Data());
    TH1D *hAllEvts = (TH1D*) gROOT->FindObject(allEvts->Data());

    TChain *ch = new TChain("tree");
    ch -> Add( Form("%s" , samples.at(i).Data()) );

    MitDMSTree signal;
    signal.LoadTree(samples.at(i).Data(),"DMSTree");
    signal.InitTree(0);

    TString ofn(samples.at(i).Data());
    ofn.ReplaceAll(".root",Form("_mva_%d.root",count));
    TFile *out = TFile::Open( Form("%s" , ofn.Data() ) ,"RECREATE" );
    out->cd();
    //TTree *clone = signal.tree_->CloneTree(0);
    TTree *clone = new TTree("exampleEventsNtuple","exampleEventsNtuple");

   if(Use["BDT"] && Use["BDTG"]) {
     printf("BDT and BDTG on, exit\n");
     assert(0);
   }

    Float_t bdt;
    Float_t ann;
    Float_t lik;

    //TBranch* br_bdt = 0;
    //TBranch* br_ann = 0;
    //TBranch* br_lik = 0;

    //if(clone->GetBranchStatus(Form("bdt_monox_%s" ,suffix.Data())) != 0) clone->SetBranchStatus(Form("bdt_monox_%s" ,suffix.Data()),0);
    //if(clone->GetBranchStatus(Form("ann_monox_%s" ,suffix.Data())) != 0) clone->SetBranchStatus(Form("ann_monox_%s" ,suffix.Data()),0);
    //if(clone->GetBranchStatus(Form("lik_monox_%s" ,suffix.Data())) != 0) clone->SetBranchStatus(Form("lik_monox_%s" ,suffix.Data()),0);

    if(Use["BDT"]||Use["BDTG"]) clone->Branch(Form("bdt_monox_%s" ,suffix.Data()) , &bdt  , Form("bdt_monox_%s/F"  ,suffix.Data()) );
    if(Use["MLP"])              clone->Branch(Form("ann_monox_%s" ,suffix.Data()) , &ann  , Form("ann_monox_%s/F"  ,suffix.Data()) );
    if(Use["LikelihoodD"])      clone->Branch(Form("lik_monox_%s" ,suffix.Data()) , &lik  , Form("lik_monox_%s/F"  ,suffix.Data()) );

    //if(Use["BDTG"])         br_bdt -> SetTitle(Form("BDTG        Output monox %s" , suffix.Data()));
    //if(Use["MLP"])          br_ann -> SetTitle(Form("MLP         Output monox %s" , suffix.Data()));
    //if(Use["LikelihoodD"])  br_lik -> SetTitle(Form("LikelihoodD Output monox %s" , suffix.Data()));

    // --- Event loop

    // Prepare the event tree
    // - here the variable names have to corresponds to your tree
    // - you can use the same variables as above which is slightly faster,
    //   but of course you can use different ones and copy the values inside the event loop
    //
  
    TStopwatch sw;
    sw.Start();

    int npass   = 0;

    for (Long64_t ievt=0; ievt<signal.tree_->GetEntries();ievt++) {
    //for (Long64_t ievt=0; ievt<100;ievt++) {

      if (ievt%100000 == 0) std::cout << "--- ... Processing event: " << ievt << std::endl;

      signal.tree_->GetEntry(ievt);

      LorentzVector fJET = signal.fjet1_;
      QGtag	      = signal.fjet1QGtag_;
      tau1            = signal.fjet1Tau1_;
      tau2            = signal.fjet1Tau2_;
      tau2tau1        = signal.fjet1Tau2_/signal.fjet1Tau1_;
      jetC2b0	      = signal.fjet1C2b0_;
      jetC2b0p2       = signal.fjet1C2b0p2_;
      jetC2b0p5       = signal.fjet1C2b0p5_;
      jetC2b1	      = signal.fjet1C2b1_;
      jetC2b2	      = signal.fjet1C2b2_;
      jetQJetVol      = signal.fjet1QJetVol_;
      jetMassSDb0     = signal.fjet1MassSDb0_;
      jetMassSDb1     = signal.fjet1MassSDb1_;
      jetMassSDb2     = signal.fjet1MassSDb2_;
      jetMassSDbm1    = signal.fjet1MassSDbm1_; 
      jetMassPruned   = signal.fjet1MassPruned_;
      jetMassFiltered = signal.fjet1MassFiltered_;
      jetMassTrimmed  = signal.fjet1MassTrimmed_;
      jetMassRaw      = signal.fjet1_.M();
      jetPull         = signal.fjet1Pull_;
      jetPullAngle    = signal.fjet1PullAngle_;
      jetQGtagSub1    = signal.fjet1QGtagSub1_;
      jetQGtagSub2    = signal.fjet1QGtagSub2_;
      jetQGtagComb    = 2.*signal.fjet1QGtagSub2_+signal.fjet1QGtagSub1_;

      if (nsel == 0) { // top selection

        Bool_t jetCuts = ((signal.fjet1_.Pt() > 250 && abs(signal.fjet1_.Eta()) < 2.5 && signal.fjet1Btag_ < 0.244) ||
    			  (signal.fjet2_.Pt() > 250 && abs(signal.fjet2_.Eta()) < 2.5 && signal.fjet2Btag_ < 0.244)) &&
        		   signal.nbjets_ >= 2;
        if     (jetCuts && (signal.fjet1_.Pt() > 250 && abs(signal.fjet1_.Eta()) < 2.5 && signal.fjet1Btag_ < 0.244)) {}
        else if(jetCuts && (signal.fjet2_.Pt() > 250 && abs(signal.fjet2_.Eta()) < 2.5 && signal.fjet2Btag_ < 0.244)) {
          fJET            = signal.fjet2_;
	  QGtag	          = signal.fjet2QGtag_;
	  tau1            = signal.fjet2Tau1_;
	  tau2            = signal.fjet2Tau2_;
	  tau2tau1        = signal.fjet2Tau2_/signal.fjet2Tau1_;
	  jetC2b0         = signal.fjet2C2b0_;
	  jetC2b0p2       = signal.fjet2C2b0p2_;
	  jetC2b0p5       = signal.fjet2C2b0p5_;
	  jetC2b1	  = signal.fjet2C2b1_;
	  jetC2b2	  = signal.fjet2C2b2_;
	  jetQJetVol      = signal.fjet2QJetVol_;
	  jetMassSDb0     = signal.fjet2MassSDb0_;
	  jetMassSDb1     = signal.fjet2MassSDb1_;
	  jetMassSDb2     = signal.fjet2MassSDb2_;
	  jetMassSDbm1    = signal.fjet2MassSDbm1_; 
	  jetMassPruned   = signal.fjet2MassPruned_;
	  jetMassFiltered = signal.fjet2MassFiltered_;
	  jetMassTrimmed  = signal.fjet2MassTrimmed_;
	  jetMassRaw      = signal.fjet2_.M();
	  jetPull         = signal.fjet2Pull_;
	  jetPullAngle    = signal.fjet2PullAngle_;
	  jetQGtagSub1    = signal.fjet2QGtagSub1_;
	  jetQGtagSub2    = signal.fjet2QGtagSub2_;
	  jetQGtagComb    = 2.*signal.fjet2QGtagSub2_+signal.fjet2QGtagSub1_;
        }
      }
      frozen = gRandom->Uniform(0.000,0.001);
      npass++;
      
      if (Use["BDTG"]){
        bdt  = reader->EvaluateMVA( "BDTG method" );
      }
      if (Use["BDT"]){
        bdt  = reader->EvaluateMVA( "BDT method" );
      }
      if (Use["MLP"]){
        ann  = reader->EvaluateMVA( "MLP method" );
      }
      if (Use["LikelihoodD"]){
        lik  = reader->EvaluateMVA( "LikelihoodD method" );
      }

      clone->Fill();

    } // End main loop

    std::cout << npass << " events passing selection" << std::endl;
 
    // Get elapsed time
    sw.Stop();
    std::cout << "--- End of event loop: "; sw.Print();

    // --- write output baby

    clone->Write();
    hAllEvts->Write();
    out->Close();

    delete reader;
    
    std::cout << "==> TMVAClassificationApplication is done to sample " << ofn.Data() << endl << std::endl;
  } 
}
