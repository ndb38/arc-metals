### Documentation for Hayes QGIS Project ###
### Author: Nicholas Barber
### Date: 18/11/2019

I will detail different components of my final QGIS setup to provide context for my Python 
and MELTS models about what decisions I made about projections, coordinates, filtering, 
and geotransformation. 

#### GEOROC DOWNLOAD

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

#### Parsing with Python 


#### QGIS
# Clipping of gbm_hayes_2.gpkg

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

#### ABBREVIATIRONS IN CSV ####

test

Abbreviations: MET: METAMORPHIC ROCK; PEG: PEGMATITE; PER: MANTLE XENOLITH; PLU: PLUTONIC ROCK; 
SED: SEDIMENTARY ROCK; VOL: VOLCANIC ROCK; VEIN: VEIN; ORE: ORE; WR: WHOLE ROCK; GL: VOLCANIC GLASS; 
MIN: MINERAL / COMPONENT (INCL. GROUNDMASS); INC: INCLUSION; LEACH: LEACH; SAE: SUBAERIAL; SAQ: SUBAQUATIC; F: FRESH; 
E: EXTENSIVELY ALTERED; M: MODERATELY ALTERED; S: SLIGHTLY ALTERED; T: ALMOST TOTALLY ALTERED; 