#  Introduction to ArcMetals

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ndb38/arc-metals/master)

Welcome! This repository hosts the code, raw and processed data, models, figures, and relevant supplementary materials for our recently published manuscript: _Amphibole Control on Copper Systematics in Arcs: Insights from the Analysis of Global Datasets_. This article was accepted for publication in _Geochimica et Cosmochimica Acta_ on May 17th 2021, and published online on May 23rd 2021 under a Creative Commons license. The live, Open Access Paper, can be found here:  https://www.sciencedirect.com/science/article/pii/S0016703721003070. 

Cite it as: **add citation info when final version published**

Open Access is courtesy of the [Bill and Melinda Gates Foundation](https://www.gatesfoundation.org/about/policies-and-resources/open-access-policy), and the [Gates Cambridge Trust](https://www.gatescambridge.org)

To amplify and build upon the work we present in this paper, all the relevant code and supporting information discussed in the paper is made available here. Over the coming weeks and months, guides, walkthroughs, and interactive elements will be added to this repository to highlight the usefulness of our data compilation **ArcMetals**. Hopefully you or someone you know will find our work helpful in exploring your own geochemical datasets!

# Citing & Contacting Us

If you find any of the code, data, or examples shown here useful, we ask you cite either our paper (link above) and/or this repository. You can contact the corresponding author of the paper, me (Nick Barber) at ndb38@cam.ac.uk if you have any questions about our paper, ArcMetals, or other related bits of code. 

# Structure

This repository is broken down into a few main sections, following the main themes of the paper, as well as some related elements of ArcMetals.

## 1_arc-metals

The "homebase" of this repository. Here you can find the Python script we used to compile the relevant [GeoRoc](http://georoc.mpch-mainz.gwdg.de/georoc/) datasets, following our compilation Methods as described in the paper. This section also contains pre-compiled versions of the ArcMetals database, as well as companion databases (some sued in the paper, others not). See the README in this folder for more details about the code, data, and other relevant features. 

## 2_analysis

This section comprises the Jupyter notebooks and related python code used to perform the relevant analyses and create the requisite plots as seen in our manuscript. These notebooks are commented through and provide additional examples beyond what was discussed in the paper. 

## 3_modeling

Some of the standalone modeling scripts are posted here. 

Models currently expressed as Excel worksheets are posted here, and will eventually be converted to Python models. 

## 4_figures

Figures and subplots related to the paper (not all of these figures appear in the main paper).

## 5_examples

WORK IN PROGRESS

## 6_dash

WORK IN PROGRESS

## 7_documentation

WORK IN PROGRESS

# Work to Do
- [ ] Finish uploading raw datasets including descriptions of the ArcMetals data
- [X] Finalize and comment arc-metals code
- [ ] Finalize and comment analysis code
- [ ] Convert from Excel to Python, finalize, and comment modeling code
- [ ] Finalize and comment figures
- [ ] Upload any remaining documentation
- [ ] Develop Dash interface for ArcMetals
- [ ] Publish how-to guides on working with global arc chemical data, both using Python and QGIS
