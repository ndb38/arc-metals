 
"""
Spyder Editor

This script will take user inputs for each GeoRock arc csv desired. The csvs 
will be properly formatted by column, data type, and will have NaN's and duplicates
truncated. Then new column for arc location, arc type, and slab parameters (?)
will be added either natively in this script or post-parsing in QGIS.

The pre-downloaded set of CSV outputs from GeoRoc will be parsed using a master
dicitonary called 'dfs'. At different stages, this master compilation will be exported 
to preserve the database at different levels of filtering. 

The goal of this compilation in the end is to assess global arc chemistry as
it pertains to chalcophile systematics (Fran's Na vs. Cu story) and other 
relevant slab parameters (Sr/Y, density, thickness, water, etc.). The compilation
will form the backbone of the modeling componenet of my PhD thesis. 

Data from GeoRock - Convergent Margin Settings, Rock Type = VOL or PLU

Author: Nicholas Barber
V1 Date: September 5th 2019
Updated to V2: December 10th 2019
"""
#%% Preamble
import pandas as pd
import numpy as np
import os
from os.path import splitext, basename

#%% 1. Generate List of CSVs, Create Master Dicitonary
### First, compile list of all CSVs in one file. This list will be used in for 
### loops for cleaning
### https://stackoverflow.com/questions/9234560/find-all-csv-files-in-a-directory-using-python/12280052

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

mscifile = find_csv_filenames('GeoRock/datav3')
print(mscifile)

print('Beginning database creation...')

# Create DataFrame Dictionary
# https://stackoverflow.com/questions/52074325/read-multiple-csv-files-into-separate-dataframes-using-pandas
# Used error_bad_lines parameter to skip those lines that didn't fit into the 
# column structure of the dataset. 

dfs = {splitext(basename(fp))[0].split('_')[0] : pd.read_csv('GeoRock/datav3/{}'.format(fp), 
       error_bad_lines = False, encoding = 'Latin-1') for fp in mscifile}
print (dfs)

print('Database compiled!')

# Managing Dictionary. 
# Thankfully, dicitonary comprehension makes this way easier!
# I originally wanted to create dynamic variables. But why do that when I can 
# take advantage of dict comprehension? Now I can run for loops!
# From inspecting cols variable, Kermadec and Banda need to be dropped   

del dfs['BANDA']
del dfs['KERMADEC']

csvs = list(dfs.keys())
    
for fp in csvs:
    print(fp)
    print(dfs[fp].columns)
    print(len(dfs[fp].columns))
    
#%% 2. Column Comparison, Formatting, and Parsing

#Now edit the columns in eahc dataframe and save column names to lists

cols = []

print(cols)

#Next, going to clean up column names. Strip all white space, make all lowercase,
#replace spaces, parantheses, backslahes, and percentage signs. Will make 
#column-level data manoiuplation easier in matplotlib/seaborn

print('Beginning column formatting...')

for fp in csvs: 
    dfs[fp].columns = dfs[fp].columns.str.strip().str.lower().str.replace(' ', '_').\
        str.replace('(', '').str.replace(')', '').str.\
        replace('\t', '').str.replace('/', '_').str.replace('%','').\
        str.replace('.', '').str.replace('?','').str.replace('___', '_').\
        str.replace('__', '_')
    lst = list(dfs[fp].columns)    
    cols.append(lst)

print('Column name formatting successful')

# New columns

# dd new column 'arc_name' based on first string grouping in location column
# Also going to create combined lat/long columns averaging across the min and max
# outputs of GeoRoc 
print('Adding new columns based on location, lat/long')

cols2 = []

for fp in csvs:
    #Add new column 'arc_name' based on first string grouping in location column
    
    splitter = dfs[fp].location.str.split("/", expand = True )
    dfs[fp].loc[:,'arc_name'] = splitter[0]
    dfs[fp].loc[:,'loc2'] = splitter[1]
    dfs[fp].loc[:,'loc3'] = splitter[2]
    dfs[fp].loc[:,'loc4'] = splitter[3]

    #Add actual lat/long values. Do this by taking mean of lat_min and lat_max, etc.

    dfs[fp].loc[:,'latitude'] = dfs[fp][['latitude_min', 'latitude_max']].mean(axis = 1)

    dfs[fp].loc[:,'longitude'] = dfs[fp][['longitude_min', 'longitude_max']].mean(axis = 1)
    
    # Check Columns
    
    lst = list(dfs[fp].columns)    
    cols2.append(lst)
    
print('Columns successfully appended!')

#Removing some faulty columns and reducing database to standard set of columns
# Cu, Zn, Zr, Ti, Ni weight percent drop. Don't want such weird values 

elements = ['cuwt', 'znwt', 'tiwt', 'zrwt', 'niwt']

for fp in dfs.keys():
    for i in elements:
        if i in dfs[fp].columns:
            dfs[fp] = dfs[fp].drop(columns=[i])
            print(fp, '{} Removed'.format(i))
        else: 
            print(fp, '{} Clean'.format(i))
            
# Check new columns 

cols3 = []

for fp in dfs.keys():
    lst = list(dfs[fp].columns)    
    cols3.append(lst)
    
# Trim columns down to Cascades structurw -123 columns. REALLY IMPORTANT
#UPDATED 15/01/21 - using new reindex method as old .loc slicing doesn't work anymore
cols4 = []   
df0 = dfs['CASCADES']

for fp in dfs.keys():
    dfs[fp] = dfs[fp].reindex(dfs['CASCADES'].columns, axis = 1)
    lst = list(dfs[fp].columns)  
    cols4.append(lst)


# Check Dtypes before forcing through Filter 3 and 4

x = []
for fp in dfs.keys():
    print(fp, dfs[fp].shape, dfs[fp].dtypes.value_counts())
    lst = dfs[fp].dtypes
    x.append(lst)   

# Drop other bad columns (found later)

for fp in dfs.keys():
    dfs[fp].drop(['geological_age_prefix'], 
                 axis = 1, inplace = True)    

#Pre-filter database export 
            
print('Creating first merged DataFrame')
  
dfm = pd.concat(dfs, axis = 0, sort = False)

print('Creation successful!')

print("Exporting first database - before filtering")

dfm.to_csv('Georock/gbm_v3pref.csv')

print('Database exported!')

#%% 3. Filter 1: Keep only those records after 1960
# This filter can always be edite to provide a more stirngent or flexible timeframe
# Check Dtypes before filtering & prompt suer for input

print('First database filter being aplied - sorting out old references')
user_year = input('Specify cutoff year for filter (if input is 1950, only studies conducted during or after 1950 will be kept ). Please specify year in XXXX format: ')
user_year = int(user_year)

for fp in dfs.keys():
    print(fp, dfs[fp].dtypes.value_counts())

# Clean up Year Column - Remove 'No Years'

for fp in dfs.keys():
    dfs[fp].year = dfs[fp].year.replace('NO YEAR', np.nan)
    dfs[fp].year = dfs[fp].year.astype('float64')
    print(fp, dfs[fp].year.dtypes)

# FILTER 1: Remove all those records with a year older than user_year

for fp in dfs.keys():
    print(fp, dfs[fp].shape[0])
    dfs[fp] = dfs[fp].loc[dfs[fp]['year'] >= user_year]
    print(fp, dfs[fp].shape[0])

print('Filter 1 applied!')

# Exporting Filter 1 database

print('Creating post Filter 1 merged DataFrame')
  
dfm = pd.concat(dfs, axis = 0, sort = False)

print('Creation successful!')

print("Exporting database - filter 1 applied")

dfm.to_csv('Georock/gbm_v3f1.csv')

print('Database exported!')

#%% 4. Filter 2: Remove all records that don't have a reported analytical method
# Drop 'NOT GIVEN'
  
print('Applying Filter 2: removing records with no analytical method...')
  
for fp in dfs.keys():
    print(fp, dfs[fp].shape[0])
    print(dfs[fp].method.value_counts())
    dfs[fp] = dfs[fp].loc[dfs[fp]['method'] != 'NOT GIVEN']
    print(fp, dfs[fp].shape[0])
    
print('Filter 2 applied!')

# Exporting Filter 2 database

print('Creating post Filter 2 merged DataFrame')
  
dfm = pd.concat(dfs, axis = 0, sort = False)

print('Creation successful!')

print("Exporting database - filter 1,2 applied")

dfm.to_csv('Georock/gbm_v3f2.csv')

print('Database exported!')
#%% 5. Plots - Method Distribution Plots
#Import plotting modules and set consistent theme
import seaborn as sns; sns.set(style='darkgrid')
import matplotlib.pyplot as plt

#if else statement controlling whetehr user wants to proceed

plot_q = input('Do you want to create bar charts for eahc arc input showing method distribution? y/n?:')
plot_q = str(plot_q)

if plot_q == 'y':
    print('Creating plots...')
    for fp in dfs.keys():
        fig = plt.figure(figsize = (16,12))
        ax = dfs[fp]['method'].value_counts().plot(kind="bar")
        plt.title("Methods for {}".format(fp), pad = 20)
        fig.savefig('GeoRock/code/methods/{}_method_plot'.format(fp), dpi = 90)
    print('Plots created!')
elif plot_q == 'n':
    print('Skipping plots')
else:
    print('Incorrect input. Specify y/n. Skipping plots')

#%% 6. Main Filter Documenttation -  Handling Duplicate Samples Smartly
'''This is going to be the trickiest bit ion the whole script. I need to 
do a few things. Originally I was going to go the route of Steve Turner. 
But my approach below keeps a record of all the duplicate handling, stored
mainly in the method_f column. Unless I reduce datapoint density down to individual volcanic centres, I don't see a need to apply as vigorous a heirarchical fitler structrue as Steve did. This is a more lax approach, that after ensuring each method is onloy represented once in a given sample_name/type_of_material combination, collapses the sampe/material duplicates down to a single  record. The methods that were collapsed into this record are stored in method_f. The reported value for a given elements/chemical measure is stored in the elements named column, while the spread of the data's resulting avergaing is stored ina  paired 'x_std' column. 

'''
    
#%% 7. Filter 3: Remove Bad Methods
# Examine distribution of methods in dataframe format, as opposed to graph format
# Then, remove all those rows containing non-standard set of methods. Methods
# that are 'standard' are discussed in Method_Reports.txt. 
# Finally, combine EMP and ICPMS categories - spellings differ in dufferent DFs
    
for fp in dfs.keys():
    s_methods = {fp: dfs[fp].method.value_counts() for fp in dfs.keys()}
    dfs_methods = pd.DataFrame(s_methods[fp])

filter_methods = ['XRF', 'TIMS', 'EMP', 'EMP (EPMA)', 'EM/EMP (EPMA)', 
                  'SIMS', 'FTIR', 'ICPMS', 'ICPMS_LA', 'MC-ICPMS', 'MC_ICPMS']

print('Applying Filter 3: removing unwanted anaytical methods and their records')

#Removing unwanted methods
for fp in dfs.keys():
    # Strip whitespace first
    dfs[fp]['method'] = dfs[fp]['method'].str.lstrip()
    dfs[fp]['method'] = dfs[fp]['method'].str.rstrip()
    # Filter for standard methods only
    dfs[fp] = dfs[fp][dfs[fp]['method'].isin(filter_methods)]
    # Check
    #print(fp, dfs[fp].method.value_counts())
    # Combine all ICPMS methods
    dfs[fp]['method_f'] = dfs[fp]['method']
    dfs[fp].method_f = dfs[fp]['method_f'].replace({'MC_ICPMS':'ICPMS',
                                                    'MC-ICPMS': 'ICPMS',
                                                    'ICPMS_LA': 'ICPMS'})
    # Combine all EMP methods
    dfs[fp].method_f = dfs[fp]['method_f'].replace({'EMP (EPMA)':'EMP',
                                                    'EM/EMP (EPMA)': 'EMP'})
    # Check
    print(fp, dfs[fp].method_f.value_counts())

print('Filter 3 successful!')

# Exporting Filter 3 database

print('Creating post Filter 3 merged DataFrame')
  
dfm = pd.concat(dfs, axis = 0, sort = False)

print('Creation successful!')

print("Exporting database - filter 1,2,3 applied")

dfm.to_csv('Georock/gbm_v3f3.csv')

print('Database exported!')


#%% 8. Filter 4: averaging like methods inn Sample/Material pairs. 
'''IF Material == Material, Method_f = Method_f, and sample_name is the same
See https://stackoverflow.com/questions/37593107/pandas-groupby-efficient-conditional-aggregation, 
https://stackoverflow.com/questions/17266129/python-pandas-conditional-sum-with-groupby, 
https://stackoverflow.com/questions/36337012/how-to-get-multiple-conditional-operations-after-a-pandas-groupby
'''

for fp in dfs.keys():
    print('Before', fp, dfs[fp].shape)
#IMPORTANT: coerece datatypes
obj = list(dfs['LUZON'].select_dtypes(include = ['object']).columns)
ints = list(dfs['LUZON'].select_dtypes(include = ['number']).columns)
#coerce data types for each dataframe to be the same
for fp in dfs.keys():
    dfs[fp][obj] = dfs[fp][obj].astype(object)
    dfs[fp][ints] = dfs[fp][ints].apply(pd.to_numeric, errors = 'coerce', axis = 1)
    
mydict1 = dict.fromkeys(ints, 'mean')
mydict2 = dict.fromkeys(obj, lambda x: x.iloc[0])
    
mydict2.pop('sample_name', None)
mydict2.pop('type_of_material', None)
mydict2.pop('method_f', None)
    
#Define custom dicitonary merge function so I only pass one dicitonary to .agg()
def Merge(dict1, dict2): 
    res = {**dict2, **dict1} 
    return res
# Creaet emrged dictionary
mydict3 = Merge(mydict1,mydict2)

print('Applying Filter 4...')

for fp in dfs.keys():
    # Need to provide method to preserve categorical variables. I create a dictionary 
    # of aggregation methods depending on column dtype. This is then passed to the
    # groupby object. Start by defining list of different column types
    
    
    dfs[fp] = dfs[fp].groupby(['sample_name', 'type_of_material', 'method_f']).\
              agg(mydict3)
              
for fp in dfs.keys():
    print('After', fp, dfs[fp].shape)

print('Filter 4 successful!')

for fp in dfs.keys():
    dfs[fp].reset_index(inplace = True)
    
# Exporting Filter 4 database

print('Creating post Filter 4 merged DataFrame')
  
dfm = pd.concat(dfs, axis = 0, sort = False)

print('Creation successful!')

print("Exporting database - filter 1,2,3,4 applied")

dfm.to_csv('Georock/gbm_v3f4.csv')

print('Database exported!')

              
#%% 9. Filter 5: Reduce to one record for a given sample/material pair
''''I think I was going about this in too complex a fashion. I've done a better job this time around
controlling for things like duplication, data quality, and other errors. 
So I want my final step to be a groupby on 'sample_name' and 'type of material
Where multiple methods are combined, I want to apply the same 'mean' and '.iloc' 
I'll catch these types of combinations by concatenating method_f 
So in structure, this block and the rpeceding block will look similar'''
# Check shape
for fp in dfs.keys():
    print('Before', fp, dfs[fp].shape)

obj = list(dfs['HONSHU'].select_dtypes(include = ['object']).columns)
ints = list(dfs['HONSHU'].select_dtypes(include = ['number']).columns)
    
mydict1 = dict.fromkeys(ints, ['mean', 'std'])
mydict2 = dict.fromkeys(obj, lambda x: x.iloc[0])
    
mydict2.pop('sample_name', None)
mydict2.pop('type_of_material', None)
mydict2.update({'method_f' : ','.join})
    
#Define custom dicitonary merge function so I only pass one dicitonary to .agg()
def Merge(dict1, dict2): 
    res = {**dict2, **dict1} 
    return res
# Creaet emrged dictionary
mydict3 = Merge(mydict1,mydict2)

print('Applying Filter 5...')

#Appply groupby method         
for fp in dfs.keys():
    # Need to provide method to preserve categorical variables. I create a dictionary 
    # of aggregation methods depending on column dtype. This is then passed to the
    # groupby object. Start by defining list of different column type
    
    dfs[fp] = dfs[fp].groupby(['sample_name', 'type_of_material']).\
              agg(mydict3)             
#Check again to see how shape has changed
for fp in dfs.keys():
    print('After', fp, dfs[fp].shape)

print('Filter 5 successful!')

# Exporting Filter 5 database

print('Creating post Filter 5 merged DataFrame')
  
dfm = pd.concat(dfs, axis = 0, sort = False)

print('Creation successful!')

print("Exporting database - filter 1,2,3,4,5 applied")

dfm.to_csv('Georock/gbm_v3f5.csv')

print('Database exported!')
#%% 10. Relabel columns and according to groupby method - for _std filtering
# Rename columns to reduce MultiIndex

print('Preivous filter made a mess of column names. Going to fix them.')

for fp in dfs.keys():
    level0 = dfs[fp].columns.get_level_values(0)
    level1 = dfs[fp].columns.get_level_values(1)
    dfs[fp].columns = level0 + '_' + level1
#Reindex and rename
#Keep '_std'

for fp in dfs.keys():
    dfs[fp].reset_index(inplace = True)

for fp in dfs.keys():
    dfs[fp].columns = dfs[fp].columns.str.replace(r'_join$', '').str.\
                      replace(r'_<lambda>$', '').str.replace(r'_mean$', '')
                      
# Drop uneccessary _stds like year

for fp in dfs.keys():
    dfs[fp].drop(['year_std', 'latitude_min_std', 'longitude_min_std',
                  'latitude_max_std', 'longitude_max_std',
                  'elevation_max_std', 'age_min_std', 'age_max_std',   
                  'latitude_std', 'longitude_std'], 
                  axis = 1, inplace = True)
    
print('Column names fixed, unecessary standard deviaiton columns dropped.')

#%% 11. Merge Dicitonary for and Prep Final Export

print('Creating final database with akll filters applied...')
  
dfm = pd.concat(dfs, axis = 0, sort = False)

# Reset Concat MultiIndex
# Need to remove index columns created by concatenation (name of dataframe in dfs &
# index of records in that sub-dataframe) from the index columns
print(dfm.index)

dfm.reset_index(inplace = True)

# Remove Sub DataFrame Name and Rename Extra Index
print(dfm.columns)

dfm.drop(columns ='level_0', inplace = True)
dfm = dfm.rename(columns ={'level_1':'arc_index'})
dfm.index.name = 'index'

print('Database created!')

#%% 12. Filter 6: Remove all rows and columns ending in _std 
#Create iterable list with _std endings

chemistry = dfm.columns[42::]

chemistry = list(chemistry)

chemistry = [i for i in chemistry if  ('_std' in i)]

#Filter 6 - Remove all rows with a std > 0

#Replace empty fields with NaN
dfm.replace(r'^\s*$', np.nan, regex=True)

print('Removing rows that averaged across methods...')

#Drop rows containg _std values
for c in chemistry:
    dfm = dfm[dfm[c].isnull()]

print('Rows removed!')

# Drop all _std columns

print('Removing all standard deviation columns...')
    
dfm = dfm[dfm.columns.drop(list(dfm.filter(regex='_std')))]

print('Columns removed!')

# Export to CSVs

print("Exporting final databse (Filters 1-6)")

dfm.to_csv('Georock/gbm_v3f6_test.csv')

print('Database_6 exported!')
