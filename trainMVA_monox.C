// @(#)root/tmva $Id: trainMVA_smurf.C,v 1.17 2013/09/16 13:31:01 ceballos Exp $
/**********************************************************************************
 * Project   : TMVA - a ROOT-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Root Macro: TMVAClassification                                                 *
 *                                                                                *
 * This macro provides examples for the training and testing of the               *
 * TMVA classifiers.                                                              *
 *                                                                                *
 * As input data is used a toy-MC sample consisting of four Gaussian-distributed  *
 * and linearly correlated input variables.                                       *
 *                                                                                *
 * The methods to be used can be switched on and off by means of booleans, or     *
 * via the prompt command, for example:                                           *
 *                                                                                *
 *    root -l ./TMVAClassification.C\(\"Fisher,Likelihood\"\)                     *
 *                                                                                *
 * (note that the backslashes are mandatory)                                      *
 * If no method given, a default set of classifiers is used.                      *
 *                                                                                *
 * The output file "TMVA.root" can be analysed with the use of dedicated          *
 * macros (simply say: root -l <macro.C>), which can be conveniently              *
 * invoked through a GUI that will appear at the end of the run of this macro.    *
 * Launch the GUI via the command:                                                *
 *                                                                                *
 *    root -l ./TMVAGui.C                                                         *
 *                                                                                *
 **********************************************************************************/

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <set>

#include "TChain.h"
#include "TMath.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TChainElement.h"

#include "MitMonoJet/macros/TMVAGui.C"

#include "Math/LorentzVector.h"
#include "Math/VectorUtil.h"

#include "MitMonoJet/Core/MitDMSTree.h"
#include "MitMonoJet/macros/functions.C"

#if not defined(__CINT__) || defined(__MAKECINT__)
// needs to be included when makecint runs (ACLIC)
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#endif

// nsel == 0 (top), 1 (Wjets), 3(Zjets), 5 (gamma+jets)

void trainMVA_monox(
 UInt_t  nselS           = 0,			   
 UInt_t  nselB           = 3,			   
 TString sigInputFile    = "/afs/cern.ch/work/c/ceballos/mva/samples/ttlj.root",
 TString bgdInputFile    = "/afs/cern.ch/work/c/ceballos/mva/samples/zll.root",
 TString outTag          = "default",
 TString myMethodList    = "BDTG,MLP,LikelihoodD",
 TString myVarList       = "QGtag,tau1,tau2,tau2tau1,jetC2b0,jetC2b0p2,jetC2b0p5,jetC2b1,jetC2b2,jetQJetVol,jetMassSDb0,jetMassSDb1,jetMassSDb2,jetMassSDbm1,jetMassPruned,jetMassFiltered,jetMassTrimmed,jetMassRaw,jetPull,jetPullAngle,jetQGtagSub1,jetQGtagSub2,jetQGtagComb"
 )
{
  // The explicit loading of the shared libTMVA is done in TMVAlogon.C, defined in .rootrc
  // if you use your private .rootrc, or run from a different directory, please copy the
  // corresponding lines from .rootrc

  // methods to be processed can be given as an argument; use format:
  //
  // mylinux~> root -l TMVAClassification.C\(\"myMethod1,myMethod2,myMethod3\"\)
  //
  // if you like to use a method via the plugin mechanism, we recommend using
  //
  // mylinux~> root -l TMVAClassification.C\(\"P_myMethod\"\)
  // (an example is given for using the BDT as plugin (see below),
  // but of course the real application is when you write your own
  // method based)
 
  //-------------------------------------------------------------------------------------
  // define event selection
  //
  // current selection is: WW selection, except anti b-tagging and soft muon veto
  //
  // The event selection is applied below, see: 
  // SIGNAL EVENT SELECTION and BACKGROUND EVENT SELECTION
  // if you want to apply any additional selection this needs to be implemented below
  //-------------------------------------------------------------------------------------

  //-----------------------------------------------------
  // choose which variables to include in MVA training
  //-----------------------------------------------------
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

  TCut sel = "";

  //---------------------------------------------------------------
  // This loads the library
  TMVA::Tools::Instance();

  // Default MVA methods to be trained + tested
  std::map<std::string,int> Use;

  // --- Cut optimisation
  Use["Cuts"]            = 1;
  Use["CutsD"]           = 1;
  Use["CutsPCA"]         = 0;
  Use["CutsGA"]          = 0;
  Use["CutsSA"]          = 0;
  // 
  // --- 1-dimensional likelihood ("naive Bayes estimator")
  Use["Likelihood"]      = 1;
  Use["LikelihoodD"]     = 0; // the "D" extension indicates decorrelated input variables (see option strings)
  Use["LikelihoodPCA"]   = 1; // the "PCA" extension indicates PCA-transformed input variables (see option strings)
  Use["LikelihoodKDE"]   = 0;
  Use["LikelihoodMIX"]   = 0;
  //
  // --- Mutidimensional likelihood and Nearest-Neighbour methods
  Use["PDERS"]           = 1;
  Use["PDERSD"]          = 0;
  Use["PDERSPCA"]        = 0;
  Use["PDEFoam"]         = 1;
  Use["PDEFoamBoost"]    = 0; // uses generalised MVA method boosting
  Use["KNN"]             = 1; // k-nearest neighbour method
  //
  // --- Linear Discriminant Analysis
  Use["LD"]              = 1; // Linear Discriminant identical to Fisher
  Use["Fisher"]          = 0;
  Use["FisherG"]         = 0;
  Use["BoostedFisher"]   = 0; // uses generalised MVA method boosting
  Use["HMatrix"]         = 0;
  //
  // --- Function Discriminant analysis
  Use["FDA_GA"]          = 1; // minimisation of user-defined function using Genetics Algorithm
  Use["FDA_SA"]          = 0;
  Use["FDA_MC"]          = 0;
  Use["FDA_MT"]          = 0;
  Use["FDA_GAMT"]        = 0;
  Use["FDA_MCMT"]        = 0;
  //
  // --- Neural Networks (all are feed-forward Multilayer Perceptrons)
  Use["MLP"]             = 0; // Recommended ANN
  Use["MLPBFGS"]         = 0; // Recommended ANN with optional training method
  Use["MLPBNN"]          = 1; // Recommended ANN with BFGS training method and bayesian regulator
  Use["CFMlpANN"]        = 0; // Depreciated ANN from ALEPH
  Use["TMlpANN"]         = 0; // ROOT's own ANN
  //
  // --- Support Vector Machine 
  Use["SVM"]             = 1;
  // 
  // --- Boosted Decision Trees
  Use["BDT"]             = 1; // uses Adaptive Boost
  Use["BDTG"]            = 0; // uses Gradient Boost
  Use["BDTB"]            = 0; // uses Bagging
  Use["BDTD"]            = 0; // decorrelation + Adaptive Boost
  // 
  // --- Friedman's RuleFit method, ie, an optimised series of cuts ("rules")
  Use["RuleFit"]         = 1;
  //
  // --- multi-output MVA's
  Use["multi_BDTG"]      = 1;
  Use["multi_MLP"]       = 1;
  Use["multi_FDA_GA"]    = 0;
  //
  // ---------------------------------------------------------------

  std::cout << std::endl;
  std::cout << "==> Start TMVAClassification" << std::endl;

  // Select methods (don't look at this code - not of interest)
  if (myMethodList != "") {
    for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;

    std::vector<TString> mlist = TMVA::gTools().SplitString( myMethodList, ',' );
    for (UInt_t i=0; i<mlist.size(); i++) {
      std::string regMethod(mlist[i]);

      if (Use.find(regMethod) == Use.end()) {
        std::cout << "Method \"" << regMethod << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
        for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) std::cout << it->first << " ";
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

  // --- Here the preparation phase begins

  // Create a ROOT output file where TMVA will store ntuples, histograms, etc.
  TString outfileName = outTag + ".root";
  TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

  // Create the factory object. Later you can choose the methods
  // whose performance you'd like to investigate. The factory is 
  // the only TMVA object you have to interact with
  //
  // The first argument is the base of the name of all the
  // weightfiles in the directory weight/
  //
  // The second argument is the output file for the training results
  // All TMVA output can be suppressed by removing the "!" (not) in
  // front of the "Silent" argument in the option string
  TMVA::Factory *factory = new TMVA::Factory( outTag.Data(), outputFile,
                                              "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );

  // If you wish to modify default settings
  // (please check "src/Config.h" to see all available global options)
  //    (TMVA::gConfig().GetVariablePlotting()).fTimesRMS = 8.0;
  //    (TMVA::gConfig().GetIONames()).fWeightFileDir = "myWeightDirectory";

  // Define the input variables that shall be used for the MVA training
  // note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
  // [all types of expressions that can also be parsed by TTree::Draw( "expression" )]
  //factory->AddVariable( "myvar1 := var1+var2", 'F' );
  //factory->AddVariable( "myvar2 := var1-var2", "Expression 2", "", 'F' );
  //factory->AddVariable( "var3",                "Variable 3", "units", 'F' );
  //factory->AddVariable( "var4",                "Variable 4", "units", 'F' );

  if (mvaVar["QGtag"])  	factory->AddVariable( "QGtag",           "QGtag",	     "", 'F' );
  if (mvaVar["tau1"])	        factory->AddVariable( "tau1",	         "tau1",             "", 'F' );
  if (mvaVar["tau2"])	        factory->AddVariable( "tau2",	         "tau2",             "", 'F' );
  if (mvaVar["tau2tau1"])	factory->AddVariable( "tau2tau1",        "tau2tau1",	     "", 'F' );
  if (mvaVar["jetC2b0"])	factory->AddVariable( "jetC2b0",         "jetC2b0",	     "", 'F' );
  if (mvaVar["jetC2b0p2"])	factory->AddVariable( "jetC2b0p2",       "jetC2b0p2",	     "", 'F' );
  if (mvaVar["jetC2b0p5"])	factory->AddVariable( "jetC2b0p5",       "jetC2b0p5",	     "", 'F' );
  if (mvaVar["jetC2b1"])	factory->AddVariable( "jetC2b1",         "jetC2b1",	     "", 'F' );
  if (mvaVar["jetC2b2"])	factory->AddVariable( "jetC2b2",         "jetC2b2",	     "", 'F' );
  if (mvaVar["jetQJetVol"])	factory->AddVariable( "jetQJetVol",      "jetQJetVol",       "", 'F' );
  if (mvaVar["jetMassSDb0"])	factory->AddVariable( "jetMassSDb0",     "jetMassSDb0",      "", 'F' );
  if (mvaVar["jetMassSDb1"])	factory->AddVariable( "jetMassSDb1",     "jetMassSDb1",      "", 'F' );
  if (mvaVar["jetMassSDb2"])	factory->AddVariable( "jetMassSDb2",     "jetMassSDb2",      "", 'F' );
  if (mvaVar["jetMassSDbm1"])	factory->AddVariable( "jetMassSDbm1",    "jetMassSDbm1",     "", 'F' );
  if (mvaVar["jetMassPruned"])  factory->AddVariable( "jetMassPruned",   "jetMassPruned",    "", 'F' );
  if (mvaVar["jetMassFiltered"])factory->AddVariable( "jetMassFiltered", "jetMassFiltered",  "", 'F' );
  if (mvaVar["jetMassTrimmed"]) factory->AddVariable( "jetMassTrimmed",  "jetMassTrimmed",   "", 'F' );
  if (mvaVar["jetMassRaw"])     factory->AddVariable( "jetMassRaw",      "jetMassRaw",       "", 'F' );
  if (mvaVar["jetPull"])	factory->AddVariable( "jetPull",	 "jetPull",          "", 'F' );
  if (mvaVar["jetPullAngle"])	factory->AddVariable( "jetPullAngle",	 "jetPullAngle",     "", 'F' );
  if (mvaVar["jetQGtagSub1"])	factory->AddVariable( "jetQGtagSub1",	 "jetQGtagSub1",     "", 'F' );
  if (mvaVar["jetQGtagSub2"])	factory->AddVariable( "jetQGtagSub2",	 "jetQGtagSub2",     "", 'F' );
  if (mvaVar["jetQGtagComb"])	factory->AddVariable( "jetQGtagComb",	 "jetQGtagComb",     "", 'F' );
  if (mvaVar["frozen"])	        factory->AddVariable( "frozen",	         "frozen",           "", 'F' );

  int nVariablesTemp = 0;

  if (mvaVar["QGtag"])  	  { cout << "Adding variable to MVA training: QGtag"           << endl; nVariablesTemp++; }
  if (mvaVar["tau1"])	          { cout << "Adding variable to MVA training: tau1"            << endl; nVariablesTemp++; }
  if (mvaVar["tau2"])	          { cout << "Adding variable to MVA training: tau2"            << endl; nVariablesTemp++; }
  if (mvaVar["tau2tau1"])	  { cout << "Adding variable to MVA training: tau2tau1"        << endl; nVariablesTemp++; }
  if (mvaVar["jetC2b0"])	  { cout << "Adding variable to MVA training: jetC2b0"         << endl; nVariablesTemp++; }
  if (mvaVar["jetC2b0p2"])	  { cout << "Adding variable to MVA training: jetC2b0p2"       << endl; nVariablesTemp++; }
  if (mvaVar["jetC2b0p5"])	  { cout << "Adding variable to MVA training: jetC2b0p5"       << endl; nVariablesTemp++; }
  if (mvaVar["jetC2b1"])	  { cout << "Adding variable to MVA training: jetC2b1"         << endl; nVariablesTemp++; }
  if (mvaVar["jetC2b2"])	  { cout << "Adding variable to MVA training: jetC2b2"         << endl; nVariablesTemp++; }
  if (mvaVar["jetQJetVol"])	  { cout << "Adding variable to MVA training: jetQJetVol"      << endl; nVariablesTemp++; }
  if (mvaVar["jetMassSDb0"])	  { cout << "Adding variable to MVA training: jetMassSDb0"     << endl; nVariablesTemp++; }
  if (mvaVar["jetMassSDb1"])	  { cout << "Adding variable to MVA training: jetMassSDb1"     << endl; nVariablesTemp++; }
  if (mvaVar["jetMassSDb2"])	  { cout << "Adding variable to MVA training: jetMassSDb2"     << endl; nVariablesTemp++; }
  if (mvaVar["jetMassSDbm1"])	  { cout << "Adding variable to MVA training: jetMassSDbm1"    << endl; nVariablesTemp++; }
  if (mvaVar["jetMassPruned"])    { cout << "Adding variable to MVA training: jetMassPruned"   << endl; nVariablesTemp++; }
  if (mvaVar["jetMassFiltered"])  { cout << "Adding variable to MVA training: jetMassFiltered" << endl; nVariablesTemp++; }
  if (mvaVar["jetMassTrimmed"])   { cout << "Adding variable to MVA training: jetMassTrimmed"  << endl; nVariablesTemp++; }
  if (mvaVar["jetMassRaw"])       { cout << "Adding variable to MVA training: jetMassRaw"      << endl; nVariablesTemp++; }
  if (mvaVar["jetPull"])	  { cout << "Adding variable to MVA training: jetPull"         << endl; nVariablesTemp++; }
  if (mvaVar["jetPullAngle"])	  { cout << "Adding variable to MVA training: jetPullAngle"    << endl; nVariablesTemp++; }
  if (mvaVar["jetQGtagSub1"])	  { cout << "Adding variable to MVA training: jetQGtagSub1"    << endl; nVariablesTemp++; }
  if (mvaVar["jetQGtagSub2"])	  { cout << "Adding variable to MVA training: jetQGtagSub2"    << endl; nVariablesTemp++; }
  if (mvaVar["jetQGtagComb"])	  { cout << "Adding variable to MVA training: jetQGtagComb"    << endl; nVariablesTemp++; }
  if (mvaVar["frozen"])	          { cout << "Adding variable to MVA training: frozen"          << endl; nVariablesTemp++; }

  const unsigned int nVariables = nVariablesTemp;
  cout << "Using " << nVariables << " variables for MVA training" << endl;

  // You can add so-called "Spectator variables", which are not used in the MVA training,
  // but will appear in the final "TestTree" produced by TMVA. This TestTree will contain the
  // input variables, the response values of all trained MVAs, and the spectator variables
  //factory->AddSpectator( "njets",  "Event Type", "units", 'F' );
  //factory->AddSpectator( "mydilmass := ",  "Spectator 2", "units", 'F' );

  MitDMSTree signal;
  signal.LoadTree(sigInputFile.Data(),"DMSTree");
  signal.InitTree(0);

  MitDMSTree background;
  background.LoadTree(bgdInputFile.Data(),"DMSTree");
  background.InitTree(0);

  // global event weights per tree (see below for setting event-wise weights)
  //Double_t signalWeight     = 1.0;
  //Double_t backgroundWeight = 1.0;
   
  // You can add an arbitrary number of signal or background trees
  //factory->AddSignalTree    ( signal,     signalWeight     );
  //factory->AddBackgroundTree( background, backgroundWeight );
      
  // To give different trees for training and testing, do as follows:
  //    factory->AddSignalTree( signalTrainingTree, signalTrainWeight, "Training" );
  //    factory->AddSignalTree( signalTestTree,     signalTestWeight,  "Test" );

  // Use the following code instead of the above two or four lines to add signal and background
  // training and test events "by hand"
  // NOTE that in this case one should not give expressions (such as "var1+var2") in the input
  //      variable definition, but simply compute the expression before adding the event
  //
  
  std::vector<Double_t> vars( nVariables );

  int nsigtrain = 0;
  int nsigtest  = 0;
  int nbkgtrain = 0;
  int nbkgtest  = 0;

  for(UInt_t ientry = 0; ientry <signal.tree_->GetEntries(); ientry++){
    signal.tree_->GetEntry(ientry);

    LorentzVector fJET        = signal.fjet1_;
    Double_t fjetQGtag        = signal.fjet1QGtag_;
    Double_t fjetPartonId     = signal.fjet1PartonId_;  
    Double_t fjetTau1	      = signal.fjet1Tau1_;
    Double_t fjetTau2	      = signal.fjet1Tau2_;
    Double_t fjetC2b0	      = signal.fjet1C2b0_;
    Double_t fjetC2b0p2       = signal.fjet1C2b0p2_;
    Double_t fjetC2b0p5       = signal.fjet1C2b0p5_;
    Double_t fjetC2b1	      = signal.fjet1C2b1_;
    Double_t fjetC2b2	      = signal.fjet1C2b2_;
    Double_t fjetQJetVol      = signal.fjet1QJetVol_;
    Double_t fjetMassSDb0     = signal.fjet1MassSDb0_;
    Double_t fjetMassSDb1     = signal.fjet1MassSDb1_;
    Double_t fjetMassSDb2     = signal.fjet1MassSDb2_;
    Double_t fjetMassSDbm1    = signal.fjet1MassSDbm1_;	
    Double_t fjetMassPruned   = signal.fjet1MassPruned_;
    Double_t fjetMassFiltered = signal.fjet1MassFiltered_;
    Double_t fjetMassTrimmed  = signal.fjet1MassTrimmed_;
    Double_t fjetPull         = signal.fjet1Pull_;
    Double_t fjetPullAngle    = signal.fjet1PullAngle_;
    Double_t fjetQGtagSub1    = signal.fjet1QGtagSub1_;
    Double_t fjetQGtagSub2    = signal.fjet1QGtagSub2_;
    Double_t fjetQGtagComb    = 2.*signal.fjet1QGtagSub2_+signal.fjet1QGtagSub1_;

    Bool_t preselCuts = (signal.preselWord_ & (1<<0)) && (signal.trigger_ & (1<<2)) && (signal.HLTmatch_ & (1<<1));
    Bool_t leptonCuts = signal.lep1_.Pt() > 30 && abs(signal.lep1_.Eta()) < 2.1 && abs(signal.lid1_) == 13 && signal.nlep_ == 1;
    Bool_t jetCuts    = ((signal.fjet1_.Pt() > 250 && abs(signal.fjet1_.Eta()) < 2.5 && signal.fjet1Btag_ < 0.244) ||
    			 (signal.fjet2_.Pt() > 250 && abs(signal.fjet2_.Eta()) < 2.5 && signal.fjet2Btag_ < 0.244)) &&
        		  signal.nbjets_ >= 2;

    LorentzVector leptonSystem = signal.lep1_;
    if(nselS == 3) leptonSystem = signal.lep1_+signal.lep2_;

    if     (nselS == 0) { // top selection

      leptonCuts = leptonCuts && deltaR(fJET.phi(),fJET.eta(),leptonSystem.phi(),leptonSystem.eta()) > 0.3;

      if     (jetCuts && (signal.fjet1_.Pt() > 250 && abs(signal.fjet1_.Eta()) < 2.5 && signal.fjet1Btag_ < 0.244)) {}
      else if(jetCuts && (signal.fjet2_.Pt() > 250 && abs(signal.fjet2_.Eta()) < 2.5 && signal.fjet2Btag_ < 0.244)) {
        fJET		 = signal.fjet2_;
    	fjetQGtag	 = signal.fjet2QGtag_;
    	fjetPartonId	 = signal.fjet2PartonId_;  
        fjetTau1	 = signal.fjet2Tau1_;
        fjetTau2	 = signal.fjet2Tau2_;
        fjetC2b0	 = signal.fjet2C2b0_;
        fjetC2b0p2	 = signal.fjet2C2b0p2_;
        fjetC2b0p5	 = signal.fjet2C2b0p5_;
        fjetC2b1	 = signal.fjet2C2b1_;
        fjetC2b2	 = signal.fjet2C2b2_;
        fjetQJetVol	 = signal.fjet2QJetVol_;
        fjetMassSDb0	 = signal.fjet2MassSDb0_;
        fjetMassSDb1	 = signal.fjet2MassSDb1_;
        fjetMassSDb2	 = signal.fjet2MassSDb2_;
        fjetMassSDbm1	 = signal.fjet2MassSDbm1_;
        fjetMassPruned   = signal.fjet2MassPruned_;
        fjetMassFiltered = signal.fjet2MassFiltered_;
        fjetMassTrimmed  = signal.fjet2MassTrimmed_;
        fjetPull         = signal.fjet2Pull_;
        fjetPullAngle    = signal.fjet2PullAngle_;
        fjetQGtagSub1    = signal.fjet2QGtagSub1_;
        fjetQGtagSub2    = signal.fjet2QGtagSub2_;
        fjetQGtagComb    = 2.*signal.fjet2QGtagSub2_+signal.fjet2QGtagSub1_;
      }
    }
    else if(nselS == 1) { // Wjets selection
      preselCuts = (signal.preselWord_ & (1<<1)) && (signal.trigger_ & (1<<2)) && (signal.HLTmatch_ & (1<<1));
      leptonCuts = signal.lep1_.Pt() > 30 && abs(signal.lep1_.Eta()) < 2.1 && abs(signal.lid1_) == 13 && signal.nlep_ == 1;
      jetCuts	 = signal.fjet1_.Pt() > 250 && abs(signal.fjet1_.Eta()) < 2.5 && signal.nbjets_ == 0;

      leptonCuts = leptonCuts && deltaR(fJET.phi(),fJET.eta(),leptonSystem.phi(),leptonSystem.eta()) > 0.3;

    }
    else if(nselS == 3) { // Zjets selection
      preselCuts = (signal.preselWord_ & (1<<2)) && (signal.trigger_ & (1<<2)) && (signal.HLTmatch_ & (1<<1));
      leptonCuts = (signal.lep1_+signal.lep2_).Pt() > 100 && signal.lep1_.Pt() > 30 && abs(signal.lep1_.Eta()) < 2.1 && abs(signal.lid1_) == 13 && signal.nlep_ == 2 && abs(signal.lid2_) == 13 && TMath::Abs(vectorSumMass(signal.lep1_.px(),signal.lep1_.py(),signal.lep1_.pz(),signal.lep2_.px(),signal.lep2_.py(),signal.lep2_.pz())-91.1876) < 15;
      jetCuts	 = signal.fjet1_.Pt() > 250 && abs(signal.fjet1_.Eta()) < 2.5;

      leptonCuts = leptonCuts && deltaR(fJET.phi(),fJET.eta(),leptonSystem.phi(),leptonSystem.eta()) > 0.3;

    }
    else if(nselS == 5) { // photon selection
      preselCuts = (signal.preselWord_ & (1<<5)) && (signal.trigger_ & (1<<3)) && (signal.HLTmatch_ & (1<<2));  
      leptonCuts = signal.nlep_ == 0 && signal.pho1_.pt() > 150;
      jetCuts	 = signal.fjet1_.Pt() > 250 && abs(signal.fjet1_.Eta()) < 2.5;  
    }

    if(!(preselCuts&&leptonCuts&&jetCuts)) continue;
    if(fJET.M() >= 140) continue;
    if(fjetPartonId != 24) continue;
    //if(fjetMassTrimmed <= 30) continue;

    int varCounter = 0;
    
    if (mvaVar["QGtag"])	   vars[varCounter++] = fjetQGtag;
    if (mvaVar["tau1"])            vars[varCounter++] = fjetTau1;
    if (mvaVar["tau2"])            vars[varCounter++] = fjetTau2;
    if (mvaVar["tau2tau1"])	   vars[varCounter++] = fjetTau2/fjetTau1;
    if (mvaVar["jetC2b0"])	   vars[varCounter++] = fjetC2b0;
    if (mvaVar["jetC2b0p2"])	   vars[varCounter++] = fjetC2b0p2;
    if (mvaVar["jetC2b0p5"])	   vars[varCounter++] = fjetC2b0p5;
    if (mvaVar["jetC2b1"])	   vars[varCounter++] = fjetC2b1;
    if (mvaVar["jetC2b2"])	   vars[varCounter++] = fjetC2b2;
    if (mvaVar["jetQJetVol"])	   vars[varCounter++] = fjetQJetVol;
    if (mvaVar["jetMassSDb0"])	   vars[varCounter++] = fjetMassSDb0;
    if (mvaVar["jetMassSDb1"])	   vars[varCounter++] = fjetMassSDb1;
    if (mvaVar["jetMassSDb2"])	   vars[varCounter++] = fjetMassSDb2;
    if (mvaVar["jetMassSDbm1"])	   vars[varCounter++] = fjetMassSDbm1;
    if (mvaVar["jetMassPruned"])   vars[varCounter++] = fjetMassPruned;
    if (mvaVar["jetMassFiltered"]) vars[varCounter++] = fjetMassFiltered;
    if (mvaVar["jetMassTrimmed"])  vars[varCounter++] = fjetMassTrimmed;
    if (mvaVar["jetMassRaw"])      vars[varCounter++] = fJET.M();
    if (mvaVar["jetPull"])         vars[varCounter++] = fjetPull;
    if (mvaVar["jetPullAngle"])    vars[varCounter++] = fjetPullAngle;
    if (mvaVar["jetQGtagSub1"])    vars[varCounter++] = fjetQGtagSub1;
    if (mvaVar["jetQGtagSub2"])    vars[varCounter++] = fjetQGtagSub2;
    if (mvaVar["jetQGtagComb"])    vars[varCounter++] = fjetQGtagComb;
    if (mvaVar["frozen"])          vars[varCounter++] = gRandom->Uniform(0.000,0.001);

    if ( gRandom->Uniform(0,1) < 0.5 ){
      factory->AddSignalTrainingEvent( vars, 1 ); nsigtrain++;
    }
    else{
      factory->AddSignalTestEvent    ( vars, 1 ); nsigtest++;
    }
  }
  
  for(UInt_t ientry = 0; ientry <background.tree_->GetEntries(); ientry++){
    background.tree_->GetEntry(ientry);

    LorentzVector fJET        = background.fjet1_;
    Double_t fjetQGtag        = background.fjet1QGtag_;
    Double_t fjetPartonId     = background.fjet1PartonId_;  
    Double_t fjetTau1	      = background.fjet1Tau1_;
    Double_t fjetTau2	      = background.fjet1Tau2_;
    Double_t fjetC2b0	      = background.fjet1C2b0_;
    Double_t fjetC2b0p2       = background.fjet1C2b0p2_;
    Double_t fjetC2b0p5       = background.fjet1C2b0p5_;
    Double_t fjetC2b1	      = background.fjet1C2b1_;
    Double_t fjetC2b2	      = background.fjet1C2b2_;
    Double_t fjetQJetVol      = background.fjet1QJetVol_;
    Double_t fjetMassSDb0     = background.fjet1MassSDb0_;
    Double_t fjetMassSDb1     = background.fjet1MassSDb1_;
    Double_t fjetMassSDb2     = background.fjet1MassSDb2_;
    Double_t fjetMassSDbm1    = background.fjet1MassSDbm1_;	
    Double_t fjetMassPruned   = background.fjet1MassPruned_;
    Double_t fjetMassFiltered = background.fjet1MassFiltered_;
    Double_t fjetMassTrimmed  = background.fjet1MassTrimmed_;
    Double_t fjetPull         = background.fjet1Pull_;
    Double_t fjetPullAngle    = background.fjet1PullAngle_;
    Double_t fjetQGtagSub1    = background.fjet1QGtagSub1_;
    Double_t fjetQGtagSub2    = background.fjet1QGtagSub2_;
    Double_t fjetQGtagComb    = 2.*background.fjet1QGtagSub2_+background.fjet1QGtagSub1_;

    Bool_t preselCuts = (background.preselWord_ & (1<<0)) && (background.trigger_ & (1<<2)) && (background.HLTmatch_ & (1<<1));
    Bool_t leptonCuts = background.lep1_.Pt() > 30 && abs(background.lep1_.Eta()) < 2.1 && abs(background.lid1_) == 13 && background.nlep_ == 1;
    Bool_t jetCuts    = ((background.fjet1_.Pt() > 250 && abs(background.fjet1_.Eta()) < 2.5 && background.fjet1Btag_ < 0.244) ||
    			 (background.fjet2_.Pt() > 250 && abs(background.fjet2_.Eta()) < 2.5 && background.fjet2Btag_ < 0.244)) &&
        		  background.nbjets_ >= 2;

    LorentzVector leptonSystem = background.lep1_;
    if(nselB == 3) leptonSystem = background.lep1_+background.lep2_;

    if     (nselB == 0) { // top selection

      leptonCuts = leptonCuts && deltaR(fJET.phi(),fJET.eta(),leptonSystem.phi(),leptonSystem.eta()) > 0.3;

      if     (jetCuts && (background.fjet1_.Pt() > 250 && abs(background.fjet1_.Eta()) < 2.5 && background.fjet1Btag_ < 0.244)) {}
      else if(jetCuts && (background.fjet2_.Pt() > 250 && abs(background.fjet2_.Eta()) < 2.5 && background.fjet2Btag_ < 0.244)) {
        fJET		 = background.fjet2_;
    	fjetQGtag	 = background.fjet2QGtag_;
    	fjetPartonId	 = background.fjet2PartonId_;  
        fjetTau1	 = background.fjet2Tau1_;
        fjetTau2	 = background.fjet2Tau2_;
        fjetC2b0	 = background.fjet2C2b0_;
        fjetC2b0p2	 = background.fjet2C2b0p2_;
        fjetC2b0p5	 = background.fjet2C2b0p5_;
        fjetC2b1	 = background.fjet2C2b1_;
        fjetC2b2	 = background.fjet2C2b2_;
        fjetQJetVol	 = background.fjet2QJetVol_;
        fjetMassSDb0	 = background.fjet2MassSDb0_;
        fjetMassSDb1	 = background.fjet2MassSDb1_;
        fjetMassSDb2	 = background.fjet2MassSDb2_;
        fjetMassSDbm1	 = background.fjet2MassSDbm1_;
        fjetMassPruned   = background.fjet2MassPruned_;
        fjetMassFiltered = background.fjet2MassFiltered_;
        fjetMassTrimmed  = background.fjet2MassTrimmed_;
	fjetPull  	 = background.fjet2Pull_;
	fjetPullAngle	 = background.fjet2PullAngle_;
	fjetQGtagSub1	 = background.fjet2QGtagSub1_;
	fjetQGtagSub2	 = background.fjet2QGtagSub2_;
	fjetQGtagComb	 = 2.*background.fjet2QGtagSub2_+background.fjet2QGtagSub1_;
      }
    }
    else if(nselB == 1) { // Wjets selection
      preselCuts = (background.preselWord_ & (1<<1)) && (background.trigger_ & (1<<2)) && (background.HLTmatch_ & (1<<1));
      leptonCuts = background.lep1_.Pt() > 30 && abs(background.lep1_.Eta()) < 2.1 && abs(background.lid1_) == 13 && background.nlep_ == 1;
      jetCuts	 = background.fjet1_.Pt() > 250 && abs(background.fjet1_.Eta()) < 2.5 && background.nbjets_ == 0;

      leptonCuts = leptonCuts && deltaR(fJET.phi(),fJET.eta(),leptonSystem.phi(),leptonSystem.eta()) > 0.3;

    }
    else if(nselB == 3) { // Zjets selection
      preselCuts = (background.preselWord_ & (1<<2)) && (background.trigger_ & (1<<2)) && (background.HLTmatch_ & (1<<1));
      leptonCuts = (background.lep1_+background.lep2_).Pt() > 100 && background.lep1_.Pt() > 30 && abs(background.lep1_.Eta()) < 2.1 && abs(background.lid1_) == 13 && background.nlep_ == 2 && abs(background.lid2_) == 13 && TMath::Abs(vectorSumMass(background.lep1_.px(),background.lep1_.py(),background.lep1_.pz(),background.lep2_.px(),background.lep2_.py(),background.lep2_.pz())-91.1876) < 15;
      jetCuts	 = background.fjet1_.Pt() > 250 && abs(background.fjet1_.Eta()) < 2.5;

      leptonCuts = leptonCuts && deltaR(fJET.phi(),fJET.eta(),leptonSystem.phi(),leptonSystem.eta()) > 0.3;

    }
    else if(nselB == 5) { // photon selection
      preselCuts = (background.preselWord_ & (1<<5)) && (background.trigger_ & (1<<3)) && (background.HLTmatch_ & (1<<2));  
      leptonCuts = background.nlep_ == 0 && background.pho1_.pt() > 150;
      jetCuts	 = background.fjet1_.Pt() > 250 && abs(background.fjet1_.Eta()) < 2.5;  
    }

    if(!(preselCuts&&leptonCuts&&jetCuts)) continue;
    if(fJET.M() >= 140) continue;
    if(fjetPartonId == 24) continue;
    //if(fjetMassTrimmed <= 30) continue;

    int varCounter = 0;

    if (mvaVar["QGtag"])	   vars[varCounter++] = fjetQGtag;
    if (mvaVar["tau1"])            vars[varCounter++] = fjetTau1;
    if (mvaVar["tau2"])            vars[varCounter++] = fjetTau2;
    if (mvaVar["tau2tau1"])	   vars[varCounter++] = fjetTau2/fjetTau1;
    if (mvaVar["jetC2b0"])	   vars[varCounter++] = fjetC2b0;
    if (mvaVar["jetC2b0p2"])	   vars[varCounter++] = fjetC2b0p2;
    if (mvaVar["jetC2b0p5"])	   vars[varCounter++] = fjetC2b0p5;
    if (mvaVar["jetC2b1"])	   vars[varCounter++] = fjetC2b1;
    if (mvaVar["jetC2b2"])	   vars[varCounter++] = fjetC2b2;
    if (mvaVar["jetQJetVol"])	   vars[varCounter++] = fjetQJetVol;
    if (mvaVar["jetMassSDb0"])	   vars[varCounter++] = fjetMassSDb0;
    if (mvaVar["jetMassSDb1"])	   vars[varCounter++] = fjetMassSDb1;
    if (mvaVar["jetMassSDb2"])	   vars[varCounter++] = fjetMassSDb2;
    if (mvaVar["jetMassSDbm1"])	   vars[varCounter++] = fjetMassSDbm1;
    if (mvaVar["jetMassPruned"])   vars[varCounter++] = fjetMassPruned;
    if (mvaVar["jetMassFiltered"]) vars[varCounter++] = fjetMassFiltered;
    if (mvaVar["jetMassTrimmed"])  vars[varCounter++] = fjetMassTrimmed;
    if (mvaVar["jetMassRaw"])      vars[varCounter++] = fJET.M();
    if (mvaVar["jetPull"])         vars[varCounter++] = fjetPull;
    if (mvaVar["jetPullAngle"])    vars[varCounter++] = fjetPullAngle;
    if (mvaVar["jetQGtagSub1"])    vars[varCounter++] = fjetQGtagSub1;
    if (mvaVar["jetQGtagSub2"])    vars[varCounter++] = fjetQGtagSub2;
    if (mvaVar["jetQGtagComb"])    vars[varCounter++] = fjetQGtagComb;
    if (mvaVar["frozen"])          vars[varCounter++] = gRandom->Uniform(0.000,0.001);

    if ( gRandom->Uniform(0,1) < 0.5 ){
      factory->AddBackgroundTrainingEvent( vars, 1 ); nbkgtrain++;
    }
    else{
      factory->AddBackgroundTestEvent    ( vars, 1 ); nbkgtest++;
    }
  }

  cout << "Add signal/background events" << endl;
  cout << "train: " << nsigtrain << " / " << nbkgtrain << endl;
  cout << "test: "  << nsigtest  << " / " << nbkgtest  << endl;

  // --- end ------------------------------------------------------------
  //
  // --- end of tree registration 
   
  // Set individual event weights (the variables must exist in the original TTree)
  factory->SetSignalWeightExpression    (1);
  factory->SetBackgroundWeightExpression(1);
  cout << "Done setting weights" << endl;

  // Apply additional cuts on the signal and background samples (can be different)
  TCut mycuts = sel; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
  TCut mycutb = sel; // for example: TCut mycutb = "abs(var1)<0.5";

  // Tell the factory how to use the training and testing events
  //
  // If no numbers of events are given, half of the events in the tree are used 
  // for training, and the other half for testing:
  //    factory->PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );
  // To also specify the number of testing events, use:
  //    factory->PrepareTrainingAndTestTree( mycut,
  //                                         "NSigTrain=3000:NBkgTrain=3000:NSigTest=3000:NBkgTest=3000:SplitMode=Random:!V" );
   
  //Use random splitting
  factory->PrepareTrainingAndTestTree( mycuts, mycutb,
                                       "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V" );

  //Use alternate splitting 
  //(this is preferable since its easier to track which events were used for training, but the job crashes! need to fix this...)
  //factory->PrepareTrainingAndTestTree( mycuts, mycutb,
  //                                     "nTrain_Signal=0:nTrain_Background=0:SplitMode=Alternate:NormMode=NumEvents:!V" );

  // ---- Book MVA methods
  //
  // Please lookup the various method configuration options in the corresponding cxx files, eg:
  // src/MethoCuts.cxx, etc, or here: http://tmva.sourceforge.net/optionRef.html
  // it is possible to preset ranges in the option string in which the cut optimisation should be done:
  // "...:CutRangeMin[2]=-1:CutRangeMax[2]=1"...", where [2] is the third input variable

  // Cut optimisation
  if (Use["Cuts"])
    factory->BookMethod( TMVA::Types::kCuts, "Cuts",
                         "!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart" );

  if (Use["CutsD"])
    factory->BookMethod( TMVA::Types::kCuts, "CutsD",
                         "!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart:VarTransform=Decorrelate" );

  if (Use["CutsPCA"])
    factory->BookMethod( TMVA::Types::kCuts, "CutsPCA",
                         "!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart:VarTransform=PCA" );

  if (Use["CutsGA"])
    factory->BookMethod( TMVA::Types::kCuts, "CutsGA",
                         "H:!V:FitMethod=GA:CutRangeMin[0]=-10:CutRangeMax[0]=10:VarProp[1]=FMax:EffSel:Steps=30:Cycles=3:PopSize=400:SC_steps=10:SC_rate=5:SC_factor=0.95" );

  if (Use["CutsSA"])
    factory->BookMethod( TMVA::Types::kCuts, "CutsSA",
                         "!H:!V:FitMethod=SA:EffSel:MaxCalls=150000:KernelTemp=IncAdaptive:InitialTemp=1e+6:MinTemp=1e-6:Eps=1e-10:UseDefaultScale" );

  // Likelihood ("naive Bayes estimator")
  if (Use["Likelihood"])
    factory->BookMethod( TMVA::Types::kLikelihood, "Likelihood",
                         "H:!V:!TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=30:NSmoothBkg[0]=30:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50" );

  // Decorrelated likelihood
  if (Use["LikelihoodD"])
    factory->BookMethod( TMVA::Types::kLikelihood, "LikelihoodD",
    			 "!H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmooth=5:NAvEvtPerBin=50:VarTransform=Decorrelate" );

  // PCA-transformed likelihood
  if (Use["LikelihoodPCA"])
    factory->BookMethod( TMVA::Types::kLikelihood, "LikelihoodPCA",
                         "!H:!V:!TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmooth=5:NAvEvtPerBin=50:VarTransform=PCA" ); 

  // Use a kernel density estimator to approximate the PDFs
  if (Use["LikelihoodKDE"])
    factory->BookMethod( TMVA::Types::kLikelihood, "LikelihoodKDE",
                         "!H:!V:!TransformOutput:PDFInterpol=KDE:KDEtype=Gauss:KDEiter=Adaptive:KDEFineFactor=0.3:KDEborder=None:NAvEvtPerBin=50" ); 

  // Use a variable-dependent mix of splines and kernel density estimator
  if (Use["LikelihoodMIX"])
    factory->BookMethod( TMVA::Types::kLikelihood, "LikelihoodMIX",
                         "!H:!V:!TransformOutput:PDFInterpolSig[0]=KDE:PDFInterpolBkg[0]=KDE:PDFInterpolSig[1]=KDE:PDFInterpolBkg[1]=KDE:PDFInterpolSig[2]=Spline2:PDFInterpolBkg[2]=Spline2:PDFInterpolSig[3]=Spline2:PDFInterpolBkg[3]=Spline2:KDEtype=Gauss:KDEiter=Nonadaptive:KDEborder=None:NAvEvtPerBin=50" ); 

  // Test the multi-dimensional probability density estimator
  // here are the options strings for the MinMax and RMS methods, respectively:
  //      "!H:!V:VolumeRangeMode=MinMax:DeltaFrac=0.2:KernelEstimator=Gauss:GaussSigma=0.3" );
  //      "!H:!V:VolumeRangeMode=RMS:DeltaFrac=3:KernelEstimator=Gauss:GaussSigma=0.3" );
  if (Use["PDERS"])
    factory->BookMethod( TMVA::Types::kPDERS, "PDERS",
                         "!H:!V:NormTree=T:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600" );

  if (Use["PDERSD"])
    factory->BookMethod( TMVA::Types::kPDERS, "PDERSD",
                         "!H:!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:VarTransform=Decorrelate" );

  if (Use["PDERSPCA"])
    factory->BookMethod( TMVA::Types::kPDERS, "PDERSPCA",
                         "!H:!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:VarTransform=PCA" );

  // Multi-dimensional likelihood estimator using self-adapting phase-space binning
  if (Use["PDEFoam"])
    factory->BookMethod( TMVA::Types::kPDEFoam, "PDEFoam",
                         "H:!V:SigBgSeparate=F:TailCut=0.001:VolFrac=0.0333:nActiveCells=500:nSampl=2000:nBin=5:Nmin=100:Kernel=None:Compress=T" );

  if (Use["PDEFoamBoost"])
    factory->BookMethod( TMVA::Types::kPDEFoam, "PDEFoamBoost",
                         "!H:!V:Boost_Num=30:Boost_Transform=linear:SigBgSeparate=F:MaxDepth=4:UseYesNoCell=T:DTLogic=MisClassificationError:FillFoamWithOrigWeights=F:TailCut=0:nActiveCells=500:nBin=20:Nmin=400:Kernel=None:Compress=T" );

  // K-Nearest Neighbour classifier (KNN)
  if (Use["KNN"])
    factory->BookMethod( TMVA::Types::kKNN, "KNN",
                         "H:nkNN=31:ScaleFrac=0.8:SigmaFact=1.0:Kernel=Gaus:UseKernel=F:UseWeight=T:!Trim" );

  // H-Matrix (chi2-squared) method
  if (Use["HMatrix"])
    factory->BookMethod( TMVA::Types::kHMatrix, "HMatrix", "!H:!V" );

  // Linear discriminant (same as Fisher discriminant)
  if (Use["LD"])
    factory->BookMethod( TMVA::Types::kLD, "LD", "H:!V:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" );

  // Fisher discriminant (same as LD)
  if (Use["Fisher"])
    factory->BookMethod( TMVA::Types::kFisher, "Fisher", "H:!V:Fisher:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=40:NsmoothMVAPdf=10" );

  // Fisher with Gauss-transformed input variables
  if (Use["FisherG"])
    factory->BookMethod( TMVA::Types::kFisher, "FisherG", "H:!V:VarTransform=Gauss" );

  // Composite classifier: ensemble (tree) of boosted Fisher classifiers
  if (Use["BoostedFisher"])
    factory->BookMethod( TMVA::Types::kFisher, "BoostedFisher", 
                         "H:!V:Boost_Num=20:Boost_Transform=log:Boost_Type=AdaBoost:Boost_AdaBoostBeta=0.2" );

  // Function discrimination analysis (FDA) -- test of various fitters - the recommended one is Minuit (or GA or SA)
  if (Use["FDA_MC"])
    factory->BookMethod( TMVA::Types::kFDA, "FDA_MC",
                         "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MC:SampleSize=100000:Sigma=0.1" );

  if (Use["FDA_GA"]) // can also use Simulated Annealing (SA) algorithm (see Cuts_SA options])
    factory->BookMethod( TMVA::Types::kFDA, "FDA_GA",
                         "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=GA:PopSize=300:Cycles=3:Steps=20:Trim=True:SaveBestGen=1" );

  if (Use["FDA_SA"]) // can also use Simulated Annealing (SA) algorithm (see Cuts_SA options])
    factory->BookMethod( TMVA::Types::kFDA, "FDA_SA",
                         "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=SA:MaxCalls=15000:KernelTemp=IncAdaptive:InitialTemp=1e+6:MinTemp=1e-6:Eps=1e-10:UseDefaultScale" );

  if (Use["FDA_MT"])
    factory->BookMethod( TMVA::Types::kFDA, "FDA_MT",
                         "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=2:UseImprove:UseMinos:SetBatch" );

  if (Use["FDA_GAMT"])
    factory->BookMethod( TMVA::Types::kFDA, "FDA_GAMT",
                         "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=GA:Converger=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=0:!UseImprove:!UseMinos:SetBatch:Cycles=1:PopSize=5:Steps=5:Trim" );

  if (Use["FDA_MCMT"])
    factory->BookMethod( TMVA::Types::kFDA, "FDA_MCMT",
                         "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MC:Converger=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=0:!UseImprove:!UseMinos:SetBatch:SampleSize=20" );

  // TMVA ANN: MLP (recommended ANN) -- all ANNs in TMVA are Multilayer Perceptrons
  if (Use["MLP"])
    factory->BookMethod( TMVA::Types::kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:!UseRegulator" );

  if (Use["MLPBFGS"])
    factory->BookMethod( TMVA::Types::kMLP, "MLPBFGS", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:TrainingMethod=BFGS:!UseRegulator" );

  if (Use["MLPBNN"])
    factory->BookMethod( TMVA::Types::kMLP, "MLPBNN", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:TrainingMethod=BFGS:UseRegulator" ); // BFGS training with bayesian regulators

  // CF(Clermont-Ferrand)ANN
  if (Use["CFMlpANN"])
    factory->BookMethod( TMVA::Types::kCFMlpANN, "CFMlpANN", "!H:!V:NCycles=2000:HiddenLayers=N+1,N"  ); // n_cycles:#nodes:#nodes:...  

  // Tmlp(Root)ANN
  if (Use["TMlpANN"])
    factory->BookMethod( TMVA::Types::kTMlpANN, "TMlpANN", "!H:!V:NCycles=200:HiddenLayers=N+1,N:LearningMethod=BFGS:ValidationFraction=0.3"  ); // n_cycles:#nodes:#nodes:...

  // Support Vector Machine
  if (Use["SVM"])
    factory->BookMethod( TMVA::Types::kSVM, "SVM", "Gamma=0.25:Tol=0.001:VarTransform=Norm" );

  // Boosted Decision Trees
  if (Use["BDTG"]) // Gradient Boost
    factory->BookMethod( TMVA::Types::kBDT, "BDTG",
                         "!H:!V:NTrees=200::BoostType=Grad:Shrinkage=0.1:UseBaggedGrad=F:nCuts=2000:NNodesMax=10000:MaxDepth=5:UseYesNoLeaf=F:nEventsMin=200:NegWeightTreatment=IgnoreNegWeights" );

  if (Use["BDT"])  // Adaptive Boost
    factory->BookMethod(TMVA::Types::kBDT,"BDT",
			 "!H:!V:NTrees=2000:BoostType=Grad:Shrinkage=0.10:UseBaggedGrad:GradBaggingFraction=0.5:nCuts=2000:NNodesMax=5:VarTransform=Decorrelate:NegWeightTreatment=IgnoreNegWeights");

  if (Use["BDTB"]) // Bagging
    factory->BookMethod( TMVA::Types::kBDT, "BDTB",
                         "!H:!V:NTrees=2000:BoostType=Bagging:SeparationType=GiniIndex:nCuts=2000:NNodesMax=100000:PruneMethod=NoPruning:NegWeightTreatment=IgnoreNegWeights" );

  if (Use["BDTD"]) // Decorrelation + Adaptive Boost
    factory->BookMethod(TMVA::Types::kBDT,"BDTD",
                    "!H:!V:NTrees=2000:BoostType=AdaBoost:SeparationType=GiniIndex:nCuts=2000:NNodesMax=100000:PruneMethod=CostComplexity:PruneStrength=25.0:VarTransform=Decorrelate:NegWeightTreatment=IgnoreNegWeights");

  // RuleFit -- TMVA implementation of Friedman's method
  if (Use["RuleFit"])
    factory->BookMethod( TMVA::Types::kRuleFit, "RuleFit",
                         "H:!V:RuleFitModule=RFTMVA:Model=ModRuleLinear:MinImp=0.001:RuleMinDist=0.001:NTrees=20:fEventsMin=0.01:fEventsMax=0.5:GDTau=-1.0:GDTauPrec=0.01:GDStep=0.01:GDNSteps=10000:GDErrScale=1.02" );

   
  // For an example of the category classifier usage, see: TMVAClassificationCategory

  // --------------------------------------------------------------------------------------------------

  // ---- Now you can optimize the setting (configuration) of the MVAs using the set of training events

  // factory->OptimizeAllMethods("SigEffAt001","Scan");
  // factory->OptimizeAllMethods("ROCIntegral","GA");

  // --------------------------------------------------------------------------------------------------

  // ---- Now you can tell the factory to train, test, and evaluate the MVAs
  
  // Train MVAs using the set of training events
  factory->TrainAllMethods();
  
  // ---- Evaluate all MVAs using the set of test events
  factory->TestAllMethods();
  
  // ----- Evaluate and compare performance of all configured MVAs
  factory->EvaluateAllMethods();
  
  // --------------------------------------------------------------

  // Save the output
  outputFile->Close();

  std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
  std::cout << "==> TMVAClassification is done!" << std::endl;
  
  delete factory;

  // Launch the GUI for the root macros
  if (!gROOT->IsBatch()) TMVAGui( outfileName );
}
