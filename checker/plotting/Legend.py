import ROOT
import itertools

# Some convenience function to easily iterate over the parts of the collections


# Needed if importing this script from another script in case TMultiGraphs are used
#ROOT.SetMemoryPolicy(ROOT.kMemoryStrict)


# Start a bit right of the Yaxis and above the Xaxis to not overlap with the ticks
start, stop = 0.18, 0.89
x_width, y_width = 0.3, 0.2
PLACES = [(start, stop - y_width, start + x_width, stop),  # top left opt
          (start, start, start + x_width, start + y_width),  # bottom left opt
          (stop - x_width, stop - y_width, stop, stop),  # top right opt
          (stop - x_width, start, stop, start + y_width),  # bottom right opt
          (stop - x_width, 0.5 - y_width / 2, stop, 0.5 + y_width / 2),  # right
          (start, 0.5 - y_width / 2, start + x_width, 0.5 + y_width / 2)]  # left


def transform_to_user(canvas, x1, y1, x2, y2):
    """
    Transforms from Pad coordinates to User coordinates.

    This can probably be replaced by using the built-in conversion commands.
    """
    xstart = canvas.GetX1()
    xlength = canvas.GetX2() - xstart
    xlow = xlength * x1 + xstart
    xhigh = xlength * x2 + xstart
    if canvas.GetLogx():
        xlow = 10**xlow
        xhigh = 10**xhigh

    ystart = canvas.GetY1()
    ylength = canvas.GetY2() - ystart
    ylow = ylength * y1 + ystart
    yhigh = ylength * y2 + ystart
    if canvas.GetLogy():
        ylow = 10**ylow
        yhigh = 10**yhigh

    return xlow, ylow, xhigh, yhigh


def overlap_h(hist, x1, y1, x2, y2):
    xlow = hist.FindFixBin(x1)
    xhigh = hist.FindFixBin(x2)
    for i in range(xlow, xhigh + 1):
        val = hist.GetBinContent(i)
        # Values
        if y1 <= val <= y2:
            return True
        # Errors
        if val + hist.GetBinErrorUp(i) > y1 and val - hist.GetBinErrorLow(i) < y2:
            # print "Overlap with histo", hist.GetName(), "at bin", i
            return True
    return False


def overlap_rect(rect1, rect2):
    """Do the two rectangles overlap?"""
    if rect1[0] > rect2[2] or rect1[2] < rect2[0]:
        return False
    if rect1[1] > rect2[3] or rect1[3] < rect2[1]:
        return False
    return True

def overlap_g(graph, x1, y1, x2, y2):
    x_values = list(graph.GetX())
    y_values = list(graph.GetY())
    x_err = list(graph.GetEX()) or [0] * len(x_values)
    y_err = list(graph.GetEY()) or [0] * len(y_values)

    for x, ex, y, ey in zip(x_values, x_err, y_values, y_err):
        # Could maybe be less conservative
        if overlap_rect((x1, y1, x2, y2), (x - ex, y - ey, x + ex, y + ey)):
            # print "Overlap with graph", graph.GetName(), "at point", (x, y)
            return True
    return False

def place_legend(canvas, x1=None, y1=None, x2=None, y2=None, header="", option="LP"):
    # If position is specified, use that
    if all(x is not None for x in (x1, x2, y1, y2)):
        return canvas.BuildLegend(x1, y1, x2, y2, header, option)

    # Make sure all objects are correctly registered
    canvas.Update()

    # Build a list of objects to check for overlaps
    objects = []
    for x in canvas.GetListOfPrimitives():
        if isinstance(x, ROOT.TH1) or isinstance(x, ROOT.TGraph):
            objects.append(x)
        elif isinstance(x, ROOT.THStack) or isinstance(x, ROOT.TMultiGraph):
            objects.extend(x)

    for place in PLACES:
        place_user = canvas.PadtoU(*place)
        # Make sure there are no overlaps
        if any(obj.Overlap(*place_user) for obj in objects):
            continue
        return canvas.BuildLegend(place[0], place[1], place[2], place[3], header, option)
    # As a fallback, use the default values, taken from TCanvas::BuildLegend
    return canvas.BuildLegend(0.5, 0.67, 0.88, 0.88, header, option)

# Monkey patch ROOT objects to make it all work
ROOT.THStack.__iter__ = lambda self: iter(self.GetHists())
ROOT.TMultiGraph.__iter__ = lambda self: iter(self.GetListOfGraphs())
ROOT.TH1.Overlap = overlap_h
ROOT.TGraph.Overlap = overlap_g
ROOT.TPad.PadtoU = transform_to_user
ROOT.TPad.PlaceLegend = place_legend
