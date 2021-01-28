'''
Reads and manipulates KNMI wind data, so it can be used in the descent trajectory.
'''


import numpy as np
import math
import scipy.stats as stats



# BRON: KONINKLIJK NEDERLANDS METEOROLOGISCH INSTITUUT (KNMI)
# Opmerking: door stationsverplaatsingen en veranderingen in waarneemmethodieken zijn deze tijdreeksen van uurwaarden mogelijk inhomogeen! Dat betekent dat deze reeks van gemeten waarden niet geschikt is voor trendanalyse. Voor studies naar klimaatverandering verwijzen we naar de gehomogeniseerde reeks maandtemperaturen van De Bilt <http://www.knmi.nl/klimatologie/onderzoeksgegevens/homogeen_260/index.html> of de Centraal Nederland Temperatuur <http://www.knmi.nl/klimatologie/onderzoeksgegevens/CNT/>.
# 
# 
# STN      LON(east)   LAT(north)     ALT(m)  NAME
# 344:         4.447       51.962      -4.30  ROTTERDAM
# 
# YYYYMMDD = datum (YYYY=jaar,MM=maand,DD=dag); 
# HH       = tijd (HH=uur, UT.12 UT=13 MET, 14 MEZT. Uurvak 05 loopt van 04.00 UT tot 5.00 UT; 
# DD       = Windrichting (in graden) gemiddeld over de laatste 10 minuten van het afgelopen uur (360=noord, 90=oost, 180=zuid, 270=west, 0=windstil 990=veranderlijk. Zie http://www.knmi.nl/kennis-en-datacentrum/achtergrond/klimatologische-brochures-en-boeken; 
# FH       = Uurgemiddelde windsnelheid (in 0.1 m/s). Zie http://www.knmi.nl/kennis-en-datacentrum/achtergrond/klimatologische-brochures-en-boeken; 
# 
# STN,YYYYMMDD,   HH,   DD,   FH

# Code to make windrose (maybe relevant at a later time): https://gist.github.com/phobson/41b41bdd157a2bcf6e14

'''
Can be used to generate wind data based on KNMI data.
'''


data_list = []
# Read data
for line in open("agents/droneAgentHelpers/KNMI_data.txt","r"):
    data_list.append(line.split(','))

# Convert string to int
data = np.zeros((len(data_list),5))
for i in range(0,len(data_list)):
    for j in range(0,5):
        data[i][j] = int(data_list[i][j])
    
wind_x = []
wind_y = []
angles = []

rows_to_delete = []

# Set windspeed of 'variable wind' & 'no wind' to 0
# Remove entries with 'variable wind'
for i in range(0,len(data)):
    if data[i, 3] == 0:
        data[i, 4] = 0
    if data[i, 3] == 990:
        rows_to_delete.append(i)
    # if data[i,3] > 179 and data[i,3] < 271:

data = np.delete(data, rows_to_delete, 0)
for i in range(0,len(data)):
    # When verifying the angles, make sure that you transpose the final result!!!

    angle = 360-data[i,3] - 90
    if angle < 0:
        angle = 360 + angle
    angle = math.radians(angle)

    wind_x.append(data[i, 4]/10 * math.cos(angle))
    wind_y.append(data[i, 4]/10 * math.sin(angle))


wind = []
for i in range(0, len(wind_x)):
    wind.append((wind_x[i], wind_y[i]))

angles = []
for w in wind:
    angles.append(math.degrees(math.atan2(w[1],w[0])))



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # plt.hist(data[:,4]/10, bins=20, density=True, alpha=0.6, color='g')
    # xmin, xmax = plt.xlim()
    # x = np.linspace(xmin, xmax, 100)
    # p = stats.norm.pdf(x, mu, std)
    # plt.plot(x, p, 'k', linewidth=2)
    # title = "Hist of 10-min average wind-speed with \n fitted normal distirbution (mu = %.2f,  std = %.2f)" % (mu, std)
    # plt.title(title)
    #
    # plt.show()

    # import statsmodels.api as sm
    import pylab

    # stats.probplot(data[:,4]/10, dist="norm", plot=pylab)

    # pylab.show()
    # # plt.hist(data[:,4]/10)
    # plt.title("Histogram of 10-min-average wind speed [m/s] 2013-2019")
    # plt.show()
    # plt.hist(data[:,3], bins=10)
    # plt.title("Wind direction (=where wind is coming from, North=0deg)")
    # plt.show()
