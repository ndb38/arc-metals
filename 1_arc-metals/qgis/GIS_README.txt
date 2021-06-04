### Documentation for Hayes QGIS Project ###
### Author: Nicholas Barber
### Date: 18/11/2019

I will detail different components of my final QGIS setup to provide context for my Python 
and MELTS models about what decisions I made about projections, coordinates, filtering, 
and geotransformation. 

# Editing gbm_v2f6.csv

Upon importing as csv file with point coordinates specified by the latitidue/longitude columns,
I need to adjust 'longitude' column values in order to project them properly. The Hayes dataset
used a longitudinal centroid of 180 degrees rather than 0, so I need to use the same approach
as I did with the Szwillus dataset to transform the center of the projected data. 

It's relatively easy to do this. I Edit the existing GeoPackage hat is projected incorrectly.

Then I use the Field Calculator to create a new field called 'long_qgis.' The expression I
use for this field calculates a new longitude:

if ("longitude" < 0 ,  "longitude"  + 360,  "longitude" )

After calculating values for the new column, I export the geoPackage as a CSV file. 

I re-import this csv, this time specifying the X field defining the coordinates as 
'long_qgis' rather than 'longitude.' Works like a charm!

# Clipping of my dataset

I decided to clip my geopackage (my original CSV file creted from compiling all GeoRock
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

***hayes_edits_2 additions

4) Izu Bonin back-arc - added this despite lack of slab info to make sure I captured chemistry 
of these critical back-arc magmas

5) Philippine fill in. Now that I have the Philippines, I wanted to make sure the geometry 
captured the right extent, especially in central and northern philippines

6) Chilean SVZ: Realized I was missing some really important southern Andean samples
based on Hayes clips. Should preserve these to make robust comparison with island arcs, 
as Andes are THE continental arc I need to contend with. 

Removes about 4000 records (total records after clip - 55801). before: 59444

# Extracting raster values

I extract values for each point in my edited and clipped GeoPackage from the following layers. 

I do so using the following Field Calculator expression: 

raster_value( 'raster_name', 1, make_point(long_qgis, latitude))

Where 'raster_name' corresponds to name of layer I'm taking the point from

1) slab_depth_total (Hayes)
2) slab_dip_total (Hayes)
3) slab_thickness_total (Hayes)
4) szw_crsthk_360 (Szwillus)
5) sedthk_360 (Litho 1.0)**

** This required more attention, as I don't just want to extract sediment pile thickness 
at a discrete raster point. I want a regional average that is more qualitative, as the 
dataset I have fro  litho1.0 is too coarse otherwise. 

SO I developed a relatively simple GIS sequence. 

I buffered my gbm_v2f6_singles.gpkg layer to 1 dgeree distance. 

Then I created a single part geometry layer version of this (by manually changing fid
values - fids all need to be different in roder to do multipart to singlepart)

Next, I used the 'Add raster values to features' SAGA tool in the processing toolbox, specifying
'Bilinear Interpolation. 

Then, I dropped all columns from the buffer_seds dataset except the fid and sedthk_360. 

Finally, I performed a Join attribute by location, with gbm_v2f6_singles as the Input, 
buffer_seds as the join layer, Predicate = intersect, fields to add = sedthk_360, join type =
Create separate feature. 

Final output called gbm_v2f6_final.gpkg/.csv

# Manual edits of 'arc_name' column

Some records had odd assignments of arc_name, so I manually edited them

Changed point fro ALEUTIAN --> Kamachatka (fid 26806)

Deleted 10+ Marianas points in the middle of the Philippines

Converted nearly 1000 Izu Bonin arc records to Honshu, given they lay on the Japanese mainland
, and for soem reason Mt. Fuji was assigned to izu, not Honshu. 

Converted around 8 Vanuatu points to Tonga

# Adding Syracuse et al. 2006 volcano H, H', Velocity, slab age, etc. (January 31st 2020)
(See their README for field details)

Wanted to do something similar for Syracuse 2006 arc convergence/slab age data that I did for
sedimentary data. However, in the case of seds_thk_360 I had a raster grid. I only had vector 
geometries to work with here from Syracuses point locations for each volcano. 

First, I cleaned the data of NULL values and standardized column data type in Excel, and removed units
row

Next, I imported based on Lon-QGIS and LaT fields. 

Then buffered to 0.1 0.1 degree distance in WGS84, which works out to a biffer distance of around 20 km. 

Finally I had to do the join, but the tricky part was making sure I didn't grab attributes from
overlapping buffer zones and therefore duplicate original SLabMetals entries. So I used SAGA's self-intersect to strip away overlapping area. This means I miss grabbing 
samples found in the overlap area, but it prevents duplication. 

Finally I Joined the self-intersected buffer layer to the gbmv2f6_final layer, calling each 
new field by the prefix 'syr_'. This new shapefile was expoeted and saved as both a GeoPackage and CSV called 
gbm_v2f6_final_syr.

#### ABBREVIATIRONS IN CSV ####

Abbreviations: MET: METAMORPHIC ROCK; PEG: PEGMATITE; PER: MANTLE XENOLITH; PLU: PLUTONIC ROCK; 
SED: SEDIMENTARY ROCK; VOL: VOLCANIC ROCK; VEIN: VEIN; ORE: ORE; WR: WHOLE ROCK; GL: VOLCANIC GLASS; 
MIN: MINERAL / COMPONENT (INCL. GROUNDMASS); INC: INCLUSION; LEACH: LEACH; SAE: SUBAERIAL; SAQ: SUBAQUATIC; F: FRESH; 
E: EXTENSIVELY ALTERED; M: MODERATELY ALTERED; S: SLIGHTLY ALTERED; T: ALMOST TOTALLY ALTERED; 