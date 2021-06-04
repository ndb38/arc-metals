# Method Report Summar before applying filter 3 in georock_parser.py
# My goal here is to note the major methods applied to obtain geochemical data, tabulate which
method (top 5 to 10) are most common, and from that determine a list of methods I'll use in my filters. 
# These summaries will be based on my method report plots (see python script for details)
# Note: All methods not listed here will be dropped. Top 10 capture vast majority of data for all arcs

# Nicholas Barber
# Dec 9th 2019

#AEGEAN

Top 10 Methods (in order from highest to lowest frequency):

XRF
ICPMS
INAA
EMP (EPMA)
AAS
SIMS
IGN
WET
TIMS
TIT

From these, I'd opt to exclude INAA, AAS, IGN, WET, TIT. Most older, and methods like WET
only useful for assessing Fe oxidation. TIT probably total ion - not super useful. INAA is an
older technique. In general I'd like to avoid it. AAS is ok.

That would leave XRF, ICPMS, EMP, SIMS, TIMS. Could opt for FTIR to be included for volatiles

# ALEUTIAN

Top 10 methods: 

XRF
EMP (EPMA)
ICPMS
TIMS
IGN
AR-AR
INAA
MC-ICPMS
DCPAES
TIMS_ID

From these, I'd opt to exclude IGN, AR-AR, INAA, MC-ICPMS, DCPAES, TIMS_ID. AR-AR only useful
for ages (not what I care about). MC-ICPMS seems like to fine a distinction; many of the ICPMS
measurements may have been done with MC attachments. Not sure what DCPAES is,but AES not preferred.

That leaves XRF, EMP, ICPMS, TIMS. Could opt for FTIR and SIMS to be included for volatiles.

# ANDEAN 

Top 10:

XRF
ICPMS
IGN
TIMS
ICPAES
EMP
INAA
AES
ICPMS_LA
K-AR

From these, I'd opt to exclude ICPAES, IGN, INAA, AES, K-AR. 

That leaves XRF, ICPMS, TIMS, EMP, ICPMS_LA. I'd be inclined to combine ICPMS_LA and ICPMS
groups. WHile they are different, they will both provide higher quality measurements of 
trace elements. And the setup isn't that different. Maybe combine MC-ICPMS as well?

# BISMARCK 

Top 10: 

XRF
ICPMS
TIMS
EMP
TIT
MS_ID
MC_ICPMS
AES
INAA
MS

From these, I'd remove TIT, MS_ID, AES, INAA, MS. MS seems like too broad a category if it 
in fact means mass spectrometry as I infer. 

That leaves XRF, ICPMS, TIMS, EMP, MC_ICPMS. I think I will fold MC_ICPMS, MC-ICPMS, and ICPMS_LA
into ICPMS to capture all these types of methods. 

# CASCADES

XRF
EMP
IGN
TIMS
ICPMS
INAA
AES
MC_ICPMS
ICPMS_LA
K-AR

Drop: IGN, INAA, AES, K-AR
Combine: ICPMS, MC_ICPMS, ICPMS_LA
Remaining: XRF, EMP, TIMS, ICPMS_tot

# CENTRAL AMERICA

Top 10:

XRF
EMP (EPMA)
ICPMS_LA
ICPMS
TIMS
IGN
HR_ICPMS
DCPAES
ICPAES/XRF
AR-AR

Drop: IGN, HR_ICPMS, DCPAES, ICPAES/XRF, AR-AR
Combine: ICPMS, ICPMS_LA
KEEP: XRF, EMP (EPMA), TIMS

# HONSHU

Top 10: 

XRF
TIMS
K-AR
ICPMS
IGN
EM/EMP (EPMA)
EMP (EPMA)
INAA
MS_ID
WET

Drop: K-AR, IGN, INAA, MS_ID, WET
Combine: EM/EMP (EPMA), EMP (EPMA)
Keep: XRF, TIMS, ICPMS

# IZU

Top 10:

EMP (EPMA)
XRF
ICPMS
ICPMS_LA
TIMS
IGN
SIMS
MS_ID
MC_ICPMS
FTIR

Drop: IGN, MS_ID
Combine: ICPMS, ICPMS_LA, MC_ICPMS
Keep: EMP (EPMA), XRF, TIMS, SIMS, FTIR

# KAMCHATKA

Top 10:

EMP (EPMA)
XRF
ICPMS
SIMS
IGN
TIMS
WET
MC_ICPMS
ICPAES
ICPMS_LA

Drop: IGN, WET, ICPAES
Combine: ICPMS, MC_ICPMS, ICPMS_LA
Keep: EMP (EPMA), XRF, SIMS, TIMS

# LESSER ANTILLES

Top 10: 

XRF
ICPMS
EMP (EPMA)
TIMS
MS_ID
IGN
ICPAES
SIMS
K-AR
ICPOES

Drop: MS_ID, IGN, ICPAES, K-AR, ICPOES
Combine: 
Keep: XRF, ICPMS, EMP (EPMA), TIMS, SIMS

# LUZON

XRF
IGN
TIMS
K-AR
ICPMS
EMP (EPMA)
AAS
ICPAES
ICPMS_LA
INAA

Drop: IGN, K-AR, AAS, ICPAES, INAA
Combine: ICPMS, ICPMS_LA
Keep: XRF, TIMS, EMP (EPMA)

# MARIANAS

Top 10: 

XRF
EMP (EPMA)
TIMS
ICPMS
IGN
AAS
AR-AR
MC_ICPMS
FTIR
ICPAES

Drop: IGN, AAS, AR-AR, ICPAES 
Combine: ICPMS, MC_ICPMS
Keep: XRF, EMP (EPMA), TIMS, FTIR

# MEXICO

Top 10:

XRF
ICPMS
IGN
EMP (EPMA)
TIMS
AAS
AR-AR
TIT
INAA
ICPAES

Drop: IGN, AAS, AR-AR, TIT, INAA, ICPAES
Combine:
Keep: XRF, ICPMS, EMP (EPMA), TIMS

# NEW ZEALAND

Top 10:

XRF
EMP (EPMA)
IGN
ICPMS_LA
SIMS
TIMS
ICPMS
FTIR
K-AR
MC_ICPMS

Drop: IGN, K-AR
Combine: ICPMS, ICPMS_LA, MC_ICPMS
Keep: XRF, EMP (EPMA), SIMS, TIMS, FTIR

# SUNDA

Top 10: 

XRF
IGN
ICPMS
EMP (EPMA)
TIMS
MS
MC_ICPMS
ICPAES
INAA
AAS

Drop: IGN, MS, ICPAES, INAA, AAS
Combine: ICPMS, MC_ICPMS
Keep: XRF, EMP (EPMA), TIMS

# TONGA

Top 10:

XRF
ICPMS
EMP (EPMA)
TIMS
IGN
MS_ID
INAA
MC_ICPMS
ICPMS_LA
AES

Drop: IGN, MS_ID, INAA, AES
Combine: ICPMS, MC_ICPMS, ICPMS_LA
Keep: XRF, EMP (EPMA), TIMS

# VANUATU

Top 10: 

EMP (EPMA)
XRF
IGN
TIMS
AES
ICPMS
AAS
ICPAES
INAA
MC_ICPMS

Drop: IGN, AES, AAS, ICPAES, INAA
Combine: ICPMS, MC_ICPMS
Keep: EMP (EPMA), XRF, TIMS