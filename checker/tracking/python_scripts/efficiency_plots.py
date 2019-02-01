#!/usr/bin/python

# Script for accessing histograms of reconstructible and
# reconstructed tracks for different tracking categories
# created by PrChecker2
#
# The efficency is calculated usig TGraphAsymmErrors
# and Bayesian error bars
#
# author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
# date:   10/2018
# 

import os,sys 
import argparse 
import ROOT
from ROOT import *
from ROOT import gStyle
from ROOT import gROOT
from ROOT import TStyle
from ROOT import gPad

sys.path.append('../../')
from plotting.LHCbStyle import *
from plotting.Legend import *

from ConfigHistos import *

def getEfficiencyHistoNames() :
    return ["eta", "p", "pt", "phi", "nPV"]
    
def getTrackers() :
    return ["Velo", "Upstream", "Forward"]
    
def getGhostHistoNames() :
    #return ["eta", "nPV"] # currently no eta information available from track
    return ["nPV"]

f = ROOT.TFile.Open("../../../output/PrCheckerPlots.root", "read")
outputfile = ROOT.TFile( "efficiency_plots.root", "recreate" )

setLHCbStyle()

efficiencyHistoDict = efficiencyHistoDict()
efficiencyHistos    = getEfficiencyHistoNames()
ghostHistos         = getGhostHistoNames()
ghostHistoDict      = ghostHistoDict()
categories          = categoriesDict()
cuts                = getCuts()
trackers            = getTrackers()

for tracker in trackers :
    outputfile.cd()
    trackerDir = outputfile.mkdir(tracker)
    trackerDir.cd()
    
    for cut in cuts[tracker]:
        cutDir = trackerDir.mkdir(cut)
        cutDir.cd()
        histoBaseName = tracker + "/" + cut + "_"

        # calculate efficiency
        for histo in efficiencyHistos:
            title = "efficiency vs. " + histo + ", " + categories[tracker][cut]["title"]
            name = "efficiency vs. " + histo
            canvas = ROOT.TCanvas(name, title)
            ROOT.gPad.SetTicks()
            # get efficiency for not electrons category
            histoName       = histoBaseName + "notElectrons_" + efficiencyHistoDict[histo]["variable"]
            print "not electrons: " + histoName
            numeratorName   = histoName + "_reconstructed"
            numerator       = f.Get(numeratorName)
            denominatorName = histoName + "_reconstructible"
            denominator     = f.Get(denominatorName)
            print numerator.GetEntries()
            print denominator.GetEntries()
            if numerator.GetEntries() == 0 or denominator.GetEntries() == 0 :
                continue
            numerator.Sumw2()
            denominator.Sumw2()
        
            g_efficiency_notElectrons = ROOT.TGraphAsymmErrors()
            g_efficiency_notElectrons.Divide(numerator, denominator, "cl=0.683 b(1,1) mode")
            g_efficiency_notElectrons.SetTitle("not electrons")
           
            # get efficiency for electrons category
            if categories[tracker][cut]["plotElectrons"] : 
                histoName       = histoBaseName + "electrons_" + efficiencyHistoDict[histo]["variable"]
                print "electrons: " + histoName
                numeratorName   = histoName + "_reconstructed"
                numerator       = f.Get(numeratorName)
                denominatorName = histoName + "_reconstructible"
                denominator     = f.Get(denominatorName)
                if numerator.GetEntries() == 0 or denominator.GetEntries() == 0 :
                     continue
                numerator.Sumw2()
                denominator.Sumw2()
                
                g_efficiency_electrons = ROOT.TGraphAsymmErrors()
                g_efficiency_electrons.Divide(numerator, denominator, "cl=0.683 b(1,1) mode")
                g_efficiency_electrons.SetTitle("electrons")
                g_efficiency_electrons.SetMarkerColor(kAzure-3)
                g_efficiency_electrons.SetLineColor(kAzure-3)
                
            # draw them both
            mg = TMultiGraph()
            mg.Add(g_efficiency_notElectrons)
            if categories[tracker][cut]["plotElectrons"] :
                mg.Add(g_efficiency_electrons)
                        
            mg.Draw("ap")
            xtitle = efficiencyHistoDict[histo]["xTitle"]
            mg.GetXaxis().SetTitle(xtitle)
            mg.GetYaxis().SetTitle("efficiency")
            mg.GetYaxis().SetRangeUser(0,1)
            
            if categories[tracker][cut]["plotElectrons"] :
                canvas.PlaceLegend()
            canvas.Write()

    # calculate ghost rate
    histoBaseName = tracker + "/"
    for histo in ghostHistos :
        trackerDir.cd()
        title = "ghost rate vs " + histo
        canvas = ROOT.TCanvas(title, title)
        ROOT.gPad.SetTicks()
        numeratorName   = histoBaseName + ghostHistoDict[histo]["variable"] + "_Ghosts"
        denominatorName = histoBaseName + ghostHistoDict[histo]["variable"] + "_Total"
        print "ghost histo: " + histoBaseName
        numerator       = f.Get(numeratorName)
        denominator     = f.Get(denominatorName)
        numerator.Sumw2()
        denominator.Sumw2()
        
        g_efficiency = ROOT.TGraphAsymmErrors()
        g_efficiency.Divide(numerator, denominator, "cl=0.683 b(1,1) mode")
        
        xtitle = ghostHistoDict[histo]["xTitle"]
        g_efficiency.GetXaxis().SetTitle(xtitle)
        g_efficiency.GetYaxis().SetTitle("ghost rate")
        g_efficiency.Draw("ap")
        
        canvas.Write()

outputfile.Write()
outputfile.Close()
f.Close()
