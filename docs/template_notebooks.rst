Template notebooks
========================

*hylite* comes bundled with a variety of template jupyter notebooks implementing common workflows for
loading, correcting, projecting and analysing hyperspectral datasets. Notebooks for data processing
and correction are stored in the *Processing* folder, while notebooks containing analysis workflows
for corrected datasets are in the *Analysis* folder.

Processing templates
-----------------------

Processing notebooks are organised into three
main categories depending on the sensor configuration: aerial (for UAV projects),
terrestrial (for ground based sensors) and core scanner (for lab scanners such as SiSu Rock).

**Scanner:**

.. toctree::
   :maxdepth: 1

   notebooks/processing/core_scanner/1/1_radiance_to_reflectance_Fenix
   notebooks/processing/core_scanner/2/2_hand_sample_extraction

**Terrestrial:**

.. toctree::
   :maxdepth: 1

   notebooks/processing/terrestrial/1/1_sensor_correction
   notebooks/processing/terrestrial/2/2_pose_estimation
   notebooks/processing/terrestrial/3/3_identify_calibration_targets

**UAV:**

.. toctree::
   :maxdepth: 1

   notebooks/processing/uav/1/1_Correct_internal_UAS
   notebooks/processing/uav/2/2_Estimate_external_UAS

Once appropriate pose/calibration data has been added to image header files using the above
notebooks, projection onto a hypercloud can then be performed:

.. toctree::
   :maxdepth: 1

   notebooks/processing/build_hypercloud
   notebooks/processing/check_hypercloud/hypercloud_QAQC


Analysis templates
----------------------

A selection of template notebooks for different analysis types can be found
in template_notebooks/analysis. These include:

.. toctree::
   :maxdepth: 1

   notebooks/analysis/load_and_plot/Load_and_plot
   notebooks/analysis/multi_feature_fitting/Multi_feature_fitting
   notebooks/analysis/mnf/mnf