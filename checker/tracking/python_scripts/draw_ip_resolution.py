import os, sys
import ROOT
import numpy as np
from array import array

sys.path.append('../../')
from plotting.LHCbStyle import *
setLHCbStyle()

edges = np.array([0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
bins = zip(edges[:-1], edges[1:])
centers = array('d', 0.5 * (edges[:-1] + edges[1:]))
widths = array('d', 0.5 * (edges[1:] - edges[:-1]))


def ipRes3D(tree, var, ptBin):
    th1 = ROOT.TH1F('h', 'h', 100, 0, 0.1)
    tree.Draw(
        var + '>>h',
        'ghost==0 && 1000./best_pt>={} && 1000./best_pt<{}'.format(
            ptBin[0], ptBin[1]), 'goff')
    return th1.GetMean(), th1.GetStdDev() / np.sqrt(th1.Integral())


def ipResFit(tree, var, ptBin):
    th1 = ROOT.TH1F('h', 'h', 100, -0.1, 0.1)
    tree.Draw(
        var + '>>h',
        'ghost==0 && 1000./best_pt>={} && 1000./best_pt<{}'.format(
            ptBin[0], ptBin[1]), 'goff')
    f = ROOT.TF1('f', 'gaus', -0.1, 0.1)
    th1.Fit(f, 'R')
    return th1.GetFunction('f').GetParameter(2), th1.GetFunction(
        'f').GetParError(2)


def makeGraph(tree, var, resFunc):
    res = np.array([resFunc(tree, var, ptBin) for ptBin in bins])
    g = ROOT.TGraphErrors(
        len(centers), centers, array('d', 1000. * res[:, 0]), widths,
        array('d', 1000. * res[:, 1]))
    g.SetName(var + '_resolution')
    return g


if __name__ == '__main__':
    plotInfo = [('ipx', 'kalman_ipx', 'velo_ipx', ipResFit, 'IP_{#it{x}}'),
                ('ipy', 'kalman_ipy', 'velo_ipy', ipResFit, 'IP_{#it{y}}'),
                ('ip3d', 'kalman_ip', 'velo_ip', ipRes3D, 'IP_{3D}')]
    inFile = ROOT.TFile('../../../output/KalmanIPCheckerOutput.root')
    inTree = inFile.Get('kalman_ip_tree')
    c1 = ROOT.TCanvas('c1', 'c1')
    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextSize(0.05)
    for info in plotInfo:
        gKalman = makeGraph(inTree, info[1], info[3])
        gVelo = makeGraph(inTree, info[2], info[3])
        fKalman = ROOT.TF1('fKalman', 'pol1', 0, 2)
        fVelo = ROOT.TF1('fVelo', 'pol1', 0, 2)
        gKalman.Fit(fKalman, 'R')
        gVelo.Fit(fVelo, 'R')
        gKalman.SetLineColor(ROOT.kCyan + 1)
        gKalman.SetMarkerColor(ROOT.kCyan + 1)
        fKalman.SetLineColor(ROOT.kCyan + 1)
        gVelo.SetLineColor(ROOT.kBlack)
        gVelo.SetMarkerColor(ROOT.kBlack)
        fVelo.SetLineColor(ROOT.kBlack)
        mg = ROOT.TMultiGraph()
        mg.Add(gKalman)
        mg.Add(gVelo)
        mg.Draw('ap')
        mg.GetHistogram().GetXaxis().SetTitle('1/#it{p}_{T} [#it{c}/GeV]')
        mg.GetHistogram().GetYaxis().SetTitle(info[4] + ' resolution [#mum]')
        mg.GetHistogram().GetXaxis().SetRangeUser(edges[0], edges[-1])
        mg.GetHistogram().GetYaxis().SetRangeUser(0, 50)
        fVelo.Draw('same')
        fKalman.Draw('same')
        legend = ROOT.TLegend(0.2, 0.92, 0.4, 0.77)
        legend.AddEntry(gVelo, 'VELO')
        legend.AddEntry(gKalman, 'Kalman')
        legend.SetFillStyle(0)
        legend.Draw('same')
        txtVelo = '#sigma_{VELO} = ' + '({:2.1f}#pm{:2.1f}) + '.format(
            fVelo.GetParameter(0),
            fVelo.GetParError(0)) + '({:2.1f}#pm{:2.1f})'.format(
                fVelo.GetParameter(1),
                fVelo.GetParError(1)) + '/#it{p}_{T} #mum'
        txtKalman = '#sigma_{Kalman} = ' + '({:2.1f}#pm{:2.1f}) + '.format(
            fKalman.GetParameter(0),
            fKalman.GetParError(0)) + '({:2.1f}#pm{:2.1f})'.format(
                fVelo.GetParameter(1),
                fVelo.GetParError(1)) + '/#it{p}_{T} #mum'
        latex.DrawLatex(0.2, 0.28, txtVelo)
        latex.DrawLatex(0.2, 0.22, txtKalman)
        c1.SaveAs(info[0] + '_resolution.pdf')
