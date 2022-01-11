"""
Created on Mon Feb 25 09:56:47 2019

@author: Callum
"""
#imports the required modules

import numpy as np
import pandas as pd
import scipy as sp
import scipy.special
import os.path

#reads in the input file

file = input("\nWhat is the name of your input file (without .csv extension), if left blank 'input.csv' will be used as default: ")

# if os.path.isfile(file) == False and file != '':
# 	print('\nThis file does not exist in the current working directory (remember not to add the .csv extension)!')
#  	file = input("\nWhat is the name of your input file (without .csv extension), if left blank 'input.csv' will be used as default: ")
# else:
if file == '':
    df = pd.read_csv('input.csv')
else:
	df = pd.read_csv(file+'.csv')

#Selects a pressure in GPa

pres = input('\nPlease enter a pressure in GPa: ')

while type(pres) != float:
	try:
		pres = float(pres)
	except ValueError:
		print('\nPlease enter a valid pressure in GPa!')
		pres = input('\nPlease enter a pressure in GPa: ')
else:
	p = float(pres)

p = pres

#Calculation parameters

#Molar ratios

xna = (df['Na2O']/30.99)/(df['SiO2']/60.08+df['TiO2']/79.9+df['Al2O3']/50.98+df['FeO']/71.85+df['MgO']/40.32+df['CaO']/56.08+df['Na2O']/30.99+df['K2O']/47.1+df['MnO']/70.94)
xmg = (df['MgO']/40.32)/(df['SiO2']/60.08+df['TiO2']/79.9+df['Al2O3']/50.98+df['FeO']/71.85+df['MgO']/40.32+df['CaO']/56.08+df['Na2O']/30.99+df['K2O']/47.1+df['MnO']/70.94)
xal = (df['Al2O3']/50.98)/(df['SiO2']/60.08+df['TiO2']/79.9+df['Al2O3']/50.98+df['FeO']/71.85+df['MgO']/40.32+df['CaO']/56.08+df['Na2O']/30.99+df['K2O']/47.1+df['MnO']/70.94)
xsi = (df['SiO2']/60.08)/(df['SiO2']/60.08+df['TiO2']/79.9+df['Al2O3']/50.98+df['FeO']/71.85+df['MgO']/40.32+df['CaO']/56.08+df['Na2O']/30.99+df['K2O']/47.1+df['MnO']/70.94)
xk = (df['K2O']/47.1)/(df['SiO2']/60.08+df['TiO2']/79.9+df['Al2O3']/50.98+df['FeO']/71.85+df['MgO']/40.32+df['CaO']/56.08+df['Na2O']/30.99+df['K2O']/47.1+df['MnO']/70.94)
xca = (df['CaO']/56.08)/(df['SiO2']/60.08+df['TiO2']/79.9+df['Al2O3']/50.98+df['FeO']/71.85+df['MgO']/40.32+df['CaO']/56.08+df['Na2O']/30.99+df['K2O']/47.1+df['MnO']/70.94)
xti = (df['TiO2']/79.9)/(df['SiO2']/60.08+df['TiO2']/79.9+df['Al2O3']/50.98+df['FeO']/71.85+df['MgO']/40.32+df['CaO']/56.08+df['Na2O']/30.99+df['K2O']/47.1+df['MnO']/70.94)
xfet = (df['FeO']/71.85)/(df['SiO2']/60.08+df['TiO2']/79.9+df['Al2O3']/50.98+df['FeO']/71.85+df['MgO']/40.32+df['CaO']/56.08+df['Na2O']/30.99+df['K2O']/47.1+df['MnO']/70.94)
xmn = (df['MnO']/70.94)/(df['SiO2']/60.08+df['TiO2']/79.9+df['Al2O3']/50.98+df['FeO']/71.85+df['MgO']/40.32+df['CaO']/56.08+df['Na2O']/30.99+df['K2O']/47.1+df['MnO']/70.94)

#Fe in sulfides and silicate

feiiicalc = df['Fe2O3']
feiitot = (df['FeO']-feiiicalc*71.85/79.85)/df['FeO']
xfeii = xfet*feiitot
sulffe = 1/(1+(df['Ni']/(df['FeO']*feiitot))*0.013+(df['Cu']/(df['FeO']*feiitot))*0.025)

#Temperature

mgnom = (df['MgO']/40.32)/(df['MgO']/40.32+df['FeO']/71.85)
t = 815.3+(265.5*mgnom)+(15.37*df['MgO'])+(8.61*df['FeO'])+(6.646*(df['Na2O']+df['K2O']))+(39.16*p)
tk = t+273

#Thermodynamic parameters

lncs = 8.77-23590/tk+(1673/tk)*((6.7*(xna+xk)+4.9*xmg+8.1*xca+8.9*(xfet+xmn)+5*xti+1.8*xal)-22.2*xti*(xfet+xmn)+7.2*(xfet*xsi))-(2.06*sp.special.erf(-7.2*(xfet+xmn)))
test = 8.77-23590/tk+(1673/tk)*(6.7*(xna+xk)+4.9*xmg+8.1*xca+8.9*(xfet+xmn)+5*xti+1.8*xal-22.2*xti*(xfet+xmn)+7.2*(xfet+xsi))
gibbs = (137778-91.666*tk+8.474*tk*np.log(tk))/(8.31441*tk)+((-291)*p+351*sp.special.erf(p))/tk
lnafes = np.log(sulffe*(1-xfeii))
lnafeo = (np.log(xfeii))+(((1-xfeii)**2)*(28870-14710*xmg+1960*xca+43300*xna+95380*xk-76880*xti)+(1-xfeii)*((-62190)*xsi+31520*xsi*xsi))/(8.31441*tk)

#SCSS calculation

lnscss = (lncs)+(gibbs)+(lnafes)-(lnafeo)
scss = np.exp(lnscss)

#Writes out the SCSS data and temperature data (if required) to output files - previous files will be overwritten unless new names are given

import csv

file2 = input("\nWhat would you like to name your output file (without .csv extension), if left blank 'output.csv' will be used as default: ")

if (file2 == ''):
	outfile = open('output.csv','w')
else:
	outfile = open(file2+'.csv', 'w')

out = csv.writer(outfile)
out.writerows(map(lambda x: [x], scss))
outfile.close()

tq = input('\nWould you like to save the calculated melt temperatures? (y/n): ')

while tq != 'y' and tq != 'n':
	print('\nPlease enter either y or n!')
	tq = input('\nWould you like to save the calculated melt temperatures? (y/n): ')
else:
	if (tq == 'y'):
		namet = input("\nWhat would you like to call the temperature file (without .csv extension), if left blank 'temperatures.csv' will be used as default: ")
		if (namet ==''):
			outfilet = open('temperatures.csv', 'w')
			out2 = csv.writer(outfilet)
			out2.writerows(map(lambda x: [x], t))
			outfilet.close()
			if (file2 == ''):
				print('\nAn output file (output.csv) with calculated SCSS values in ppm and a temperature file (temperatures.csv) with calculated melt temperatures in oC have been saved in the working directory\n')
			else:
				print('\nAn output file ('+file2+'.csv) with calculated SCSS values in ppm and a temperature file (temperatures.csv) with calculated melt temperatures in oC have been saved in the working directory\n')

		else:
			outfile = open(namet+'.csv', 'w')
			out2 = csv.writer(outfilet)
			out2.writerows(map(lambda x: [x], t))
			outfilet.close()
			if (file2 == ''):
				print('\nAn output file (output.csv) with the calculated SCSS values in ppm and a temperature file ('+namet+'.csv) with calculated melt temperatures in oC have been saved in the working directory\n')
			else:
				print('\nAn output file ('+file2+'.csv) with the calculated SCSS values in ppm and a temperature file ('+namet+'.csv) with calculated melt temperatures in oC have been saved in the working directory\n')


	else:
		if (file2 == ''):
			print('\nAn output file (output.csv) with the calculated SCSS values in ppm has been saved in the working directory\n')
		else:
			print('\nAn output file ('+file2+'.csv) with the calculated SCSS values in ppm has been saved in the working directory\n')
