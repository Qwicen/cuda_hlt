#!/usr/bin/python

# Script for obtaining the momentum resolution versus momentum
# from a 2D histogram of dp versus p
#
# author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
# date:   12/2018
# 

import os,sys 
import argparse 
import ROOT
from ROOT import *
from ROOT import gStyle
from ROOT import gROOT
from ROOT import TStyle
from ROOT import gPad

from ROOT import TCanvas, TGraph
from ROOT import gROOT
from math import sin
from array import array
 
from LHCbStyle import *

def getTrackers() :
    return ["Velo", "Upstream", "Forward"]


f = ROOT.TFile.Open("../../../output/PrCheckerPlots.root", "read")
outputfile = ROOT.TFile( "momentum_resolution.root", "recreate" )

trackers = getTrackers()

for tracker in trackers :
    outputfile.cd()
    trackerDir = outputfile.mkdir(tracker)
    trackerDir.cd()

    # get histogram
    name = tracker + "/dp_vs_p"
    histo2D = f.Get(name)

    # fit slices in p
    n = 0
    x, y = array( 'd' ), array( 'd' )
    nBinsX = histo2D.GetNbinsX()
    xAxis = histo2D.GetXaxis()
    for i in range( nBinsX ):
        histo1D = histo2D.ProjectionY("_py", i, i, "")
        histo1D.Write()
        x.append( xAxis.GetBinCenter(i) )
        y.append( histo1D.GetBinContent(i) )
        print(' i %i %f %f ' % (i,x[i],y[i]))
        n+=1

    gr = TGraph( n, x, y )

   
    name = tracker + "/momentum_resolution_vs_p"
    #grtmp.SetName(name)
    
outputfile.Write()
outputfile.Close()
f.Close()
