import matplotlib.pyplot as plt
from matplotlib import path
import numpy as np
from scipy import stats
from scipy.optimize import least_squares
import matplotlib.patches as patches
import cv2

import hylite
from hylite.reference.features import HyFeature
from hylite import HyData
from hylite.project import pix_to_ray_pano, pix_to_ray_persp


class Panel( HyData ):
    """
    A class for identifying calibration reference in images and storing
    the observed pixel radiances and known target reflectance values. This is used
    by, e.g., empirical line calibration procedures.
    """

    def __init__(self, material, radiance, **kwds):
        """
        Generic constructor. Can be of the following forms:

        *Arguments*:
          - material = a hylite.reference.Target instance containing target reflectance data for this panel.
          - radiance = either a HyImage object (which contains some reference pixels) or a
                       NxM numpy array containing radiance values for N pixels across M bands.

        *Keywords*:
         - wavelengths = wavelengths corresponding to the radiance values (if radiance is an array rather than a HyImage
                         object).
         - method = 'manual' (default) to manually pick reference in image (using an interactive plot). Can also be
                     'sobel' or 'laplace' to use the corresponding edge detectors to automatically identify the
                     calibration target (using OpenCV contouring).
         - bands = list of band indices (integer) or wavelengths (float) to use when selecting the target.
         - edge_thresh = the edge threshold (as percentile of gradient values) used when automatically identifying reference.
                         Default is 95.
         - area_thresh = the minimum area (in pixels) of candidate panels. Used to discard small/bright areas. Default is 100.
         - shrink = a shrink factor to reduce the size of reference identified automatically (and so remove dodgy pixels
                    near the target edge/frame. Default is 0.4.
         - db = If True, visualisation of the edge detection layers will be plotted for debug purposes. Default is false.
        """

        super().__init__(None)  # initialise header etc.
        self.source_image = None  # init defaults

        if isinstance(radiance, np.ndarray):  # radiance is a list of pixels

            # check and copy radiance data
            if len(radiance.shape) == 1:  # we've been given mean radiances
                radiance = radiance[np.newaxis, :]
            assert len(radiance.shape) == 2, "Error, radiance must be a 2-D array (N pixels by M bands)."
            self.data = radiance.copy()

            # check and copy wavelength data
            assert 'wavelengths' in kwds, "Error - wavelengths must be provided for pixel array."
            self.set_wavelengths(np.array(kwds["wavelengths"]))

        elif radiance.is_image():  # radiance is a hyimage

            self.source_image = radiance  # store reference to original image

            method = kwds.get("method", 'manual')  # what method to use to pick target?
            bands = kwds.get("bands", 428.0)

            # select target region
            if 'manual' in method.lower():  # pick region using interactive plot
                verts = radiance.pickPolygons(region_names=['Target'], bands=bands)[0]
                verts = np.vstack([verts, verts[0][None, :]])  # add return to first point
                self.outline = path.Path(verts)  # make matplotlib path from selected region

            else:
                db = kwds.get('db', False)  # draw plots?

                # calculate greyscale image
                if isinstance(bands, tuple) or isinstance(bands, list):
                    bands = [radiance.get_band_index(b) for b in bands]  # convert to indices
                    gray = np.sum(radiance.data[:, :, bands], axis=2) / np.nanmax(radiance.data[:, :, bands])
                else:
                    bands = radiance.get_band_index(bands)
                    gray = radiance.data[:, :, bands] / np.nanmax(radiance.data[:, :, bands])
                gray = cv2.GaussianBlur(gray, (3, 3), 0)  # apply slight blur to improve edge detection

                if db:
                    plt.figure(figsize=(20, 10))
                    plt.imshow(gray.T, cmap='gray')
                    plt.title("Greyscale")
                    plt.show()

                # extract edges
                if 'sobel' in method.lower() or 'auto' in method.lower():  # pick edges using sobel filter
                    sobelx = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 1, 0, ksize=5)  # x
                    sobely = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 0, 1, ksize=5)  # y
                    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
                    thresh = np.nanpercentile(sobel, kwds.get('edge_thresh', 95))
                    edge = sobel > thresh
                elif 'laplace' in method.lower():  # pick edges using laplace filter
                    laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
                    thresh = np.nanpercentile(laplacian, kwds.get('edge_thresh', 95))
                    edge = laplacian > thresh
                else:
                    assert False, "Error - %s is not a recognised extraction method. Try 'sobel' or 'laplace'." % method

                if db:
                    plt.figure(figsize=(20, 10))
                    plt.imshow(edge.T)
                    plt.title("Edge")
                    plt.show()

                # contour and find object contours
                _, threshold = cv2.threshold(edge.astype(np.uint8) * 254, 240, 255, cv2.THRESH_BINARY)
                _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # find brightest quadrilateral with area above threshold
                maxCtr = None
                maxBright = -1
                area_thresh = kwds.get("area_thresh", 100)
                for cnt in contours:

                    # simplify/approximate contour
                    approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)

                    # is it a quadrilateral?
                    if approx.shape[0] == 4:
                        # calculate area
                        area = cv2.contourArea(approx)
                        if area > area_thresh:  # large enough?
                            # calculate approx brightness (by summing bounding box)
                            verts = np.array([approx[:, 0, 1], approx[:, 0, 0]]).T
                            xmin, xmax = np.min(verts[..., 0]), np.max(verts[..., 0])
                            ymin, ymax = np.min(verts[..., 1]), np.max(verts[..., 1])
                            patch = gray[xmin:xmax, ymin:ymax]
                            bright = np.nanmedian(patch)

                            if db:
                                plt.imshow(gray.T, cmap='gray')
                                plt.title("Candidate panel")
                                plt.axvline(xmin, color='r')
                                plt.axvline(xmax, color='r')
                                plt.axhline(ymin, color='r')
                                plt.axhline(ymax, color='r')
                                plt.show()

                            # store?
                            if maxBright < bright:
                                maxCtr = approx
                                maxBright = bright

                # convert to maplotlib path
                verts = np.array([maxCtr[:, 0, 1], maxCtr[:, 0, 0]]).T
                verts = np.vstack([verts, verts[0][None, :]])  # add return to first point
                self.outline = path.Path(verts, closed=True)  # make matplotlib path from selected region

                # shrink to 40% of original size (to remove frame and dodgy edge pixels)
                centroid = np.mean(self.outline.vertices[1::, :], axis=0)
                verts = kwds.get("shrink", 0.4) * (self.outline.vertices - centroid) + centroid
                self.outline = path.Path(verts, closed=True)

            # calculate pixels within selected region
            xx, yy = np.meshgrid(np.arange(radiance.xdim()), np.arange(radiance.ydim()))
            xx = xx.flatten()
            yy = yy.flatten()
            points = np.vstack([xx, yy]).T  # coordinates of each pixel
            mask = self.outline.contains_points(points)  # identify points within path
            mask = mask.reshape((radiance.ydim(), radiance.xdim())).T  # reshape to pixel mask

            # extract pixel reflectance values based on this mask
            self.data = np.array([radiance.data[:, :, b][mask] for b in range(radiance.band_count())]).T

            # also copy across wavelength info
            assert radiance.has_wavelengths(), "Error - radiance image must have wavelength information."
            self.set_wavelengths(radiance.get_wavelengths())  # get wavelength data

        else:
            assert False, "Error: radiance argument must be a HyImage instance or a numpy array of pixels."

        # if we have lots of target pixels (we don't need that many), only keep top 50% [as darker ones likely result from
        # dodgy border effects
        if self.data.shape[0] > 30:
            brightness = np.nanmean(self.data, axis=1)
            mask = brightness > np.nanpercentile(brightness, 50)
            self.data = self.data[mask, :]

        # extract reflectance data from target
        target_bands = material.get_wavelengths()
        assert np.nanmin(target_bands) <= np.nanmin(
            self.get_wavelengths()), "Error - calibration range does not cover pixel range. " \
                                     "Radiance data starts at %.1f nm but calibration data starts %.1f nm." % (
                                         np.nanmin(self.get_wavelengths()), np.nanmin(target_bands))
        assert np.nanmax(target_bands) >= np.nanmax(
            self.get_wavelengths()), "Error - calibration range does not cover pixel range. " \
                                     "Radiance data ends at %.1f nm but calibration data ends %.1f nm." % (
                                         np.nanmax(self.get_wavelengths()), np.nanmax(target_bands))
        idx = [np.argmin(np.abs(target_bands - w)) for w in self.get_wavelengths()]  # matching wavelengths
        self.reflectance = material.get_reflectance()[idx]
        self.material = material

    def copy(self):
        """
        Make a deep copy of this panel instance.

        *Returns*
          - a new Panel instance.
        """
        return Panel( self.material, self.data, wavelengths=self.get_wavelengths() )

    def get_mean_radiance(self):
        """
        Calculate and return the mean radiance for each band of all the pixels in this calibration region.
        """
        return np.nanmean(self.data, axis=0)

    def get_reflectance(self):
        """
        Get the known (reference) reflectance of this panel.
        """
        return self.reflectance

    def get_normal(self, cam):
        """
        Get the normal vector of this panel by assuming its outline is square (prior to projection onto the camera).

        *Arguments*:
         - a Camera object describing the pose of the camera from which the panel is viewed.

        *Returns*:
         - norm, err = the normal vector of the panel (in world coordinates) and the residual error of the orientation
                       estimation procedure.
        """

        # get corners of panel and convert to rays
        corners = np.array([self.outline.vertices[i, :] for i in range(4)])

        if cam.is_panoramic():
            ray1 = pix_to_ray_pano(corners[0, 0], corners[0, 1], cam.fov, cam.step, cam.dims)
            ray2 = pix_to_ray_pano(corners[1, 0], corners[1, 1], cam.fov, cam.step, cam.dims)
            ray3 = pix_to_ray_pano(corners[2, 0], corners[2, 1], cam.fov, cam.step, cam.dims)
            ray4 = pix_to_ray_pano(corners[3, 0], corners[3, 1], cam.fov, cam.step, cam.dims)
        else:
            ray1 = pix_to_ray_persp(corners[0, 0], corners[0, 1], cam.fov, cam.dims)
            ray2 = pix_to_ray_persp(corners[1, 0], corners[1, 1], cam.fov, cam.dims)
            ray3 = pix_to_ray_persp(corners[2, 0], corners[2, 1], cam.fov, cam.dims)
            ray4 = pix_to_ray_persp(corners[3, 0], corners[3, 1], cam.fov, cam.dims)

        a = 1.0  # length of each square (in arbitrary coordinates)
        h = np.sqrt(2)  # length of hypot relative to sides

        def opt(x, sol=False):
            # get test depths
            z1, z2, z3, z4 = x

            # calculate edge coordinates
            A = ray1 * z1
            B = ray2 * z2
            C = ray3 * z3
            D = ray4 * z4

            # and errors with edge lengths
            AB = np.linalg.norm(B - A)
            BC = np.linalg.norm(C - B)
            CD = np.linalg.norm(D - C)
            DA = np.linalg.norm(A - D)
            AC = np.linalg.norm(C - A)
            BD = np.linalg.norm(D - B)

            if not sol:
                return [AB - a, BC - a, CD - a, DA - a, AC - h, BD - h]  # return for optimiser
            else:  # return solution (normal vector)
                AB = (B - A) / AB
                BC = (C - B) / BC
                return np.cross(AB, BC)

        # get normal vector in camera coords
        sol = least_squares(opt, (10.0, 10.0, 10.0, 10.0))
        norm = opt(sol.x, sol=True)

        # rotate to world coords
        norm = np.dot(cam.get_rotation_matrix(), norm)
        if norm[2] < 0:
            norm *= -1
        return norm

    def quick_plot(self, bands=hylite.RGB, **kwds):

        """
        Quickly plot the outline of this calibration target for quality checking etc.

        *Arguments*:
         - bands = the image bands to plot as a preview. Default is io.HyImage.RGB.
        *Keywords*:
         - keywords are passed to HyData.plot_spectra( ... ).
        """
        if self.source_image is not None: # plot including preview of panel
            fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 3]})

            # plot base image
            self.source_image.quick_plot(bands, ax=ax[0])
            ax[0].set_xticks([])
            ax[0].set_yticks([])

            # plot target on image and set extents
            patch = patches.PathPatch(self.outline, edgecolor='orange', fill=False, lw=2)
            bbox = self.outline.get_extents()
            padx = (bbox.max[0] - bbox.min[0]) * 1.5
            pady = (bbox.max[1] - bbox.min[1]) * 1.5
            ax[0].add_patch(patch)
            ax[0].set_xlim(bbox.min[0] - padx, bbox.max[0] + padx)
            ax[0].set_ylim(bbox.max[1] + pady, bbox.min[1] - pady)

            # plot spectra
            kwds['labels'] = kwds.get('labels', HyFeature.Themes.ATMOSPHERE)
            self.plot_spectra( ax=ax[1], **kwds )
        else: # no image data, just plot spectra
            kwds['labels'] = kwds.get('labels', HyFeature.Themes.ATMOSPHERE)
            fig, ax = self.plot_spectra(**kwds)

        return fig, ax

    def plot_ratio(self, ax = None):

        """
        Plots the ratio between known reflectance and observed radiance for each band in this target.

        *Arguments*:
         - ax = the axes to plot on. If None (default) then a new axes is created.
        *Returns*:
         -fig, ax = the figure and axes objects containing the plot.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 10))

        # calculate ratio
        ratio = self.get_mean_radiance() / self.get_reflectance()

        # plot
        ax.plot( self.get_wavelengths(), ratio )
        ax.set_ylabel("radiance / reflectance" )
        ax.set_xlabel("Wavelength (nm)")

        return fig, ax

class ELC(object):
    """
    Class that gathers one or more Panels and computes calculates an empirical line correction.
    """

    def __init__(self, panels):

        """
        Constructor that takes a list of Panel objects (one for each target used for the correction) and computes
        an empirical line correction.

        *Arguments*:
          - panels = a list of Panel objects defining the reflectance and radiance of each panel in the scene.
        """

        if not isinstance(panels, list):
            panels = [panels]

        self.wav = np.array(panels[0].get_wavelengths())
        for p in panels:
            assert isinstance(p, Panel), "Error - ELC panels must be instances of hylite.correct.Panel"
            assert (self.wav == np.array(
                p.get_wavelengths())).all(), 'Error - ELC panels must cover the same wavelengths'

        # compute ELC
        self.slope = np.zeros(self.wav.shape)
        self.intercept = np.zeros(self.wav.shape)
        if len(panels) == 1:  # only one panel - assume intercept = 0
            self.slope = panels[0].get_reflectance() / panels[0].get_mean_radiance()
        else:
            # calculate regression for each band
            for b, w in enumerate(self.wav):
                _x = np.array([p.get_mean_radiance()[b] for p in panels])
                _y = np.array([p.get_reflectance()[b] for p in panels])
                self.slope[b], self.intercept[b], _, _, _ = stats.linregress(_x, _y)

    def get_wavelengths(self):
        """
        Get the wavelengths for which this ELC has been calculated.
        """
        return self.wav

    def get_bad_bands(self, **kwds):

        """
        Find bands in which signal-noise ratios are amplified above a threshold (due to large correction slope).

        *Keywords*:
         - thresh = the threshold slope. Defaults to the 85th percentile.

        *Returns*:
         - a boolean numpy array containing True for bad bands and False otherwise.
        """

        thresh = kwds.get("thresh", np.nanpercentile(self.slope, 85))
        return self.slope > thresh

    def apply(self, data, **kwds):

        """
        Apply this empirical line calibration to the specified image.

        *Arguments*:
         - data = a HyData instance to correct

        *Keywords*:
         - thresh = the threshold slope. Defaults to the 90th percentile.

        *Returns*:
         - a mask containing true where the corrected values are considered reasonable - see get_bad_bands(...) for more
           details. Note that this returns the np.logical_not( self.get_bad_bands(...) ).
        """

        assert data.band_count() == len(self.slope), "Error - data has %d bands but ELC has %d" % (
        data.band_count(), len(self.slope))
        data.data *= self.slope
        data.data += self.intercept

        return np.logical_not(self.get_bad_bands(**kwds))

    def quick_plot(self, ax=None, **kwds):

        """
        Plots the correction factors (slope and intercept) computed for this ELC.

        *Arguments*:
         - ax = the axes to plot on. If None (default) then a new axes is created.
        *Keywords*:
         - thresh = the threshold to separate good vs bad correction values (see get_bad_bands(...)). Default is the
                    85th percentile of slope values.
        *Returns*:
         -fig, ax = the figure and axes objects containing the plot.

        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 10))

        # plot slope
        _x = self.get_wavelengths()
        _y1 = self.slope
        _y2 = [kwds.get("thresh", np.nanpercentile(self.slope, 85))] * len(_y1)
        ax.plot(_x, _y1, color='k', lw=1)
        ax.plot(_x, _y2, color='gray', lw=2)
        ax.fill_between(_x, _y1, [0] * len(_x), where=_y1 > _y2, facecolor='red', interpolate=True, alpha=0.3)
        ax.fill_between(_x, _y1, [0] * len(_x), where=_y1 < _y2, facecolor='green', interpolate=True, alpha=0.3)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("ELC slope")
        if not (self.intercept == 0).all():
            ax2 = ax.twinx()
            ax2.plot(_x, self.intercept, color='b')
            ax2.set_ylabel("ELC intercept")
            ax2.spines['right'].set_color('blue')
            ax2.yaxis.label.set_color('blue')
            ax2.tick_params(axis='y', colors='blue')

        return fig, ax

