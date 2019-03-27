#!/usr/bin/python

# Script for obtaining the momentum resolution versus momentum
# from a 2D histogram of dp versus p
#
# author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
# date:   12/2018
#

import os, sys
import argparse
import ROOT
from ROOT import *
from ROOT import gStyle
from ROOT import gROOT
from ROOT import TStyle
from ROOT import gPad

from array import array

from LHCbStyle import *


def getHistos():
    basedict = {
        "p": {},
        "qop": {},
    }

    basedict["p"]["name"] = "momentum_resolution_vs_p_gauss"
    basedict["p"]["title"] = "momentum resolution vs p, Gaussian fit"
    basedict["p"]["x_axis_title"] = "p[MeV/c]"
    basedict["p"]["y_axis_title"] = "#sigma_{p}/p"

    basedict["qop"]["name"] = "qop_resolution_vs_qop_gauss"
    basedict["qop"]["title"] = "q/p resolution vs q/p, Gaussian fit"
    basedict["qop"]["x_axis_title"] = "q/p[c/MeV]"
    basedict["qop"]["y_axis_title"] = "#sigma_{q/p}/(q/p)"

    return basedict


def getTrackers():
    return ["Upstream", "Forward"]


def getResolutionInSlices(histo2D, var, var_dict):
    # fit slices
    n = 0
    xFit, yFit = array('d'), array('d')
    xFitErr, yFitErr = array('d'), array('d')
    rms, rmsErr = array('d'), array('d')
    nBinsX = histo2D.GetNbinsX()
    xAxis = histo2D.GetXaxis()
    for i in range(1, nBinsX - 1):
        histo1D = histo2D.ProjectionY("_py", i, i, "")
        if histo1D.GetEntries() >= 100:
            # fit Gaussian
            if tracker == "Forward":
                g1 = ROOT.TF1("g1", "gaus", -0.05, 0.05)
            elif tracker == "Upstream":
                g1 = ROOT.TF1("g1", "gaus", -0.5, 0.5)
            histo1D.Fit(g1, "R")
            histo1D.Write()
            p = xAxis.GetBinCenter(i)
            xFit.append(p)
            sigma_p = histo1D.GetFunction("g1").GetParameter(2)
            yFit.append(sigma_p)
            xFitErr.append(0)
            delta_sigma_p = histo1D.GetFunction("g1").GetParError(2)
            yFitErr.append(delta_sigma_p)

            # get RMS of histogram
            rms.append(histo1D.GetRMS())
            rmsErr.append(histo1D.GetRMSError())

            n += 1

    if n == 0:
        return

    #name = "momentum_resolution_vs_p_gauss"
    #title = "momentum resolution vs p, Gaussian fit"
    name = var_dict[var]["name"]
    title = var_dict[var]["title"]
    canvas = ROOT.TCanvas(name, title)
    canvas.cd()
    print('{:s}: n = {:d}: '.format(tracker, n))
    gr = TGraphErrors(n, xFit, yFit, xFitErr, yFitErr)
    #histoP.Draw()
    gr.Draw("ap")

    name = tracker + name
    x_axis_title = var_dict[var]["x_axis_title"]
    y_axis_title = var_dict[var]["y_axis_title"]
    gr.GetXaxis().SetTitle(x_axis_title)
    gr.GetYaxis().SetTitle(y_axis_title)
    gr.SetTitle("")
    gr.SetName(name)

    gr.Write()
    canvas.Write()

    #name = "dp_vs_p_rms"
    #title = "dp vs p, histogram RMS"
    #canvas = ROOT.TCanvas(name, title)
    #gr = TGraphErrors( n, xFit, rms, xFitErr, rmsErr )
    #gr.Draw("ap")

    #name = tracker + "/momentum_resolution_vs_p"
    #gr.GetXaxis().SetTitle("p[MeV/c]")
    #gr.GetYaxis().SetTitle("#sigma_{p}/p")
    #gr.SetTitle("")
    #gr.SetName(name)
    #gr.Write()
    #canvas.Write()

    # overall momentum resolution
    histo1D = histo2D.ProjectionY("_py")
    histo1D.Write()
    histo1D.Fit("gaus")
    sigma_p = histo1D.GetFunction("gaus").GetParameter(2)
    delta_sigma_p = histo1D.GetFunction("gaus").GetParError(2)
    print('{:s}: sigma p = {:f} +/- {:f}'.format(tracker, sigma_p,
                                                 delta_sigma_p))


f = ROOT.TFile.Open("../../../output/PrCheckerPlots.root", "read")
outputfile = ROOT.TFile("momentum_resolution.root", "recreate")

setLHCbStyle()

trackers = getTrackers()
var_dict = getHistos()

for tracker in trackers:
    outputfile.cd()
    trackerDir = outputfile.mkdir(tracker)
    trackerDir.cd()

    # momentum resolution
    name = tracker + "/momentum_resolution"
    histo2D = f.Get(name)
    name = tracker + "/p_matched"
    histoP = f.Get(name)

    getResolutionInSlices(histo2D, "p", var_dict)

    # qop resolution
    name = tracker + "/qop_resolution"
    histo2D = f.Get(name)

    getResolutionInSlices(histo2D, "qop", var_dict)

outputfile.Write()
outputfile.Close()
f.Close()
