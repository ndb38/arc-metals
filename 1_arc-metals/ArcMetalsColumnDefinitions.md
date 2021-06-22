# Documentation for Columns in ArcMetals

Here you'll find a detailed description of each and every column in the database, for reference ind determining how to use the database. Units are given where appropriate

# First 119 columns are taken from GeoRoc. Some of these were adapted, edited, or condensed following the georock_parser.py script

`fid`: indices as a result of the compilation

`index` and `arc_index`: pythonic indices used during the compilation script

`sample_name`: sample name from publication, as recorded in GeoRoc

`type_of_mineral`: GeoRoc columns telling us whether the sample in question is a whole rock, inclusion, or glass sample

`method_f`: compilation column I made during the georock_parser.py script. It concatenates all the different analytical methods recorded in each record: ICPMS,TIMS tells us the geochemical data measured in this row was obtained with a combination of TIMS and ICPMS analyses

`citation`: reference from the paper this samples data comes from. See GeoRoc for citation details

`unique_id`: a GeoRoc unique ID for the particular sample

`location`: GeoRoc locational column. Full location, broken down into smaller and smaller sub regions, given as a string list with subsections divided by '/'

`location_comment`: GeoRoc technician comment on location - often qualitative, possibly referring to geological formation

`sampling_technique`: List of field sampling techniques used to obtain the rock, like 'OUTCROP', 'DRIL CORE,' DREDGE,' etc. 

`land_sea_sample`: Two values - 'SUBAERIAL,' or terrestrial samples, and 'SUBAQUEOUS,' or marine samples

`rock_type`: Column with singular value in ArcMetals, 'VOLCANIC ROCK,' indicating that a given sample is nominally shallow or extrusive. The Plutonic version of ArcMetals sees this columns values change to 'PLUTONIC.'

`rock_name`: Petrologically assigned name for rock, with varying degrees of detail. 

`rock_texture`: Comment on petrology of rock, noting things like grain size and texture, quality, vesicularity, crystallinity, etc.

`sample_comment`: GeoRoc flagged incidental comment on sample quality, texture, or other external hand sample level features

`geological_age`: Geological epoch/age that the sample is described to have come from. Given as a name, rather than a quantitative age

`alteration`: Comment on degree of alteration. Ranging from 'EXTENSIVELY ALTERED' to 'FRESH'

`inclusion`: Comment on kind of inclusion, if the glassy sample in question is an inclusion. Could be a fluid inclusion, melt inclusion, or other

`mineral_inclusion`: if the 'inclusion' column is a mineral inclusion, this column tells you what mineral the mineral inclusion in question is.

`hostmineral_inclusion`: the host mineral for the inclusion, whatever kind of inclusion it may be

`heating`: boolean column, where value tells you whether sample was heated for analsyes e.g. whether melt inclusion was hmogenized

`method`: original analytical column, before the compilation script produced the 'method_f' column

`comment`:

`institution`:

`arc_name`:

`loc2`:

`loc3`:

`loc4`:

`loc5`:

`year`: 

