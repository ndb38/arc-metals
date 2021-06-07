# Documentation for ArcMetals Project ###
# Author: Nicholas Barber
# Creation Date: 18/11/2019
# Last Updated: 07/06/2021

I will detail different components of my final QGIS setup to provide context for my Python models about what decisions I made about projections, coordinates, filtering, and geotransformation. **First this README provides information on the step-by-step process of creating the ArcMetals database (Sections 1-3), and then it provides a full list and inventory of the files present in this repo (Section 4)**. 

## 1. Downloading CSV files from GeoRoc

Here are the steps I took to download each Arc's GeoROC .csv file with similar formats. 
The actual shape of the dataset varied given what chemical data was available for each arc.

1) Selected 'CONVERGENT MARGIN' Query from 'GEOLOGICAL SETTING' menu on the homepage

2) Selected one of 19 arcs alphabetically as follows: AEGEAN, ALUETIAN, ANDES, BANDA, 
BISMARCK, CASCADES, CENTRAL AMERICA, HONSHU, IZU BONIN, KAMCHATKA, KERMADEC, LESSER ANTILLES,
LUZON, MARIANAS, MEXICAN, NEW ZEALAND, SUNDA, TONGA, VANUATU

3) Selected 2 Sample Selection Criteria: 
    - ROCK TYPE = 'VOLCANIC ROCK'
    - TYPE OF MATERIAL = 'WHOLE ROCK', 'VOLCANIC GLASS', 'INCLUSION'
    
4) Chemistry. Tried to select as much chemical data as possible. Below covers the elements
and isotopes selected for the database. IN all cases, I always ticked WITH ANALYTICAL METHODS.
    - MAJORS: 
    - TRACE:
    - REE: 
    - RADIOGENIC ISOTOPES: 
    - No Diseq., No Stable, No Volatiles, No Gases, No Minerals
    - AGE 
    
5) At Download page, I asked GeoRoc to compile metadata into seperate sheets (I don't think
this does much, but it had the effect of making each output look similar) by ticking
CHECK ALL

6) Downloaded CSV

7) Split initial CSV into two: (1) NAMED_ARC.csv containing all data (2) NAMED_ARC_METADATA
which contained the References and selectionc riteria for my query. Usually this metadata
is stored at the bottom of a sheet, so formatting this way meant the sheets were clean for 
further analysis. 

## 2. Parsing with Python 

Wroted the **georock_parser_v3.py** script to sort throguh these CSVs efficently, make them all fit a standard mold, and applied the six filters to the data as discussed in 

## 3. QGIS

Compilation of tectonic/geophysical data done using QGIS. See Supplementary Item 1 from main manuscript regarding QGIS methods applied. Ata  alater time, QGIS maps and guides will be provided to help you replicate what I did here

### 3.1 Clipping of gbm_hayes_2.gpkg

I decided to clip gbm_hayes_2.gpkg (my original CSV file creted from compiling all GeoRock
arc datasets, importing them as a CSV, converting to points, and applying gdal_warp to 
center over Hayes datasets) to the extent of the Hayes raster files, with a few exceptions.

1) Extended Mexican coverage to include back-arc Mexican volcanics N of Mexico City. As a 
prime example of flat subduction and likely delayed devolatization, Mexico is to valuable 
an area. The Hayes slab extents stop short of the W shallow dipping limb of the slab. 

2) Extended PNG - Extended Bismarck arc coverage to include Southern Volcanic Belt in SE
Papua. Again, back-arc data likely important for understanding role of changing mantle wedge 
structure in controlling chalcophile systematics.

3) Manus Basin - Addded coverage to back-arc between Tonga and Vanuatu - incldues Manus basin
which is another end member, this time for magnetite fractionation and Fran's
'Magnetite Crisis'

## 4. Inventory of Files in this Repository



#### ABBREVIATIRONS IN CSVs ####

Abbreviations: MET: METAMORPHIC ROCK; PEG: PEGMATITE; PER: MANTLE XENOLITH; PLU: PLUTONIC ROCK; 
SED: SEDIMENTARY ROCK; VOL: VOLCANIC ROCK; VEIN: VEIN; ORE: ORE; WR: WHOLE ROCK; GL: VOLCANIC GLASS; 
MIN: MINERAL / COMPONENT (INCL. GROUNDMASS); INC: INCLUSION; LEACH: LEACH; SAE: SUBAERIAL; SAQ: SUBAQUATIC; F: FRESH; 
E: EXTENSIVELY ALTERED; M: MODERATELY ALTERED; S: SLIGHTLY ALTERED; T: ALMOST TOTALLY ALTERED; 
