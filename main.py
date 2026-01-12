import os
import matplotlib.pyplot as plt
from function import import_pos_file, plot_orbits, cart_to_geodetic, select_dop_values, geodetic_to_cart,ecef_to_ned_matrix, compute_dop_epoch, compute_dop_time_series
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Arbeitsumgebung und Import der Daten
dir = os.getcwd()
data = os.path.join(dir, "Daten")
ecef = import_pos_file(data + "/pos_ECEF.txt")
print(ecef.head())
ecsf = import_pos_file(data + "/pos_ECSF.txt")
print(ecsf.head())

# Erdparameter
r = 6371000 # Erdradius in Metern
u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)
# Erde
x = r * np.outer(np.cos(u),np.sin(v)) # Vektorprodukt
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones_like(u), np.cos(v))

print(ecef.head())
print(ecsf.head())

# Plotted Orbits
#plot_orbits(ecef, ecsf, 1)

# Groundplot zweier Satelliten
# Auswahl bestimmter PRN
prn1 = 4
prn2 = 7
satellite_1 = ecef[ecef["satellite"] == prn1]
satellite_2 = ecef[ecef["satellite"] == prn2] 

lat1, lon1, h1 = cart_to_geodetic(satellite_1["X"].values, satellite_1["Y"].values, satellite_1["Z"].values)
lat2, lon2, h2 = cart_to_geodetic(satellite_2["X"].values, satellite_2["Y"].values, satellite_2["Z"].values)


# Entfernt Linien bei Sprüngen von über 360°
lon1 = np.unwrap(np.radians(lon1))
lon1 = np.degrees(lon1)
lon2 = np.unwrap(np.radians(lon2))
lon2 = np.degrees(lon2)

# Eigentlicher Groundplot
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False   # Oben keine Labels
gl.left_labels = False  # Links keine Labels

ax.plot(lon1, lat1, 'r' , transform=ccrs.PlateCarree(),
        label=f"PRN {prn1}")

ax.plot(lon2, lat2, 'b' , transform=ccrs.PlateCarree(),
        label=f"PRN {prn2}")

ax.set_title(f"Groundtrackplot von PRN {prn1} & {prn2}")
ax.legend()

#plt.show()

## auswahl der ECEF & ECSF Values von 900 - 1200

ecef_values, ecsf_values = select_dop_values(ecef, ecsf)


# reciever position in Graz (AUT)
graz_lat = 47.084503173828125
graz_lon = 15.421300888061523
graz_height  = 353.7
graz_cart = geodetic_to_cart(graz_lat,graz_lon,graz_height)
print(graz_cart)

#reciever position in Narvik (NOR)
narvik_lat = 68.4383796
narvik_lon = 17.4271978
narvik_height = 0 #liegt am Meer
narvik_cart = geodetic_to_cart(narvik_lat,narvik_lon,narvik_height)
print(narvik_cart)

# --------------------------
# DOP-Berechnung für Graz
# --------------------------
mask_angles = [0, 5, 10, 15]
for mask in mask_angles:
    dop_df = compute_dop_time_series(
        ecef_values,
        graz_lat, graz_lon, graz_cart,
        mask_angle_deg=mask
    )
    # PDOP, HDOP, VDOP Plot
    dop_df[["PDOP","HDOP","VDOP"]].plot(title=f"DOP-Zeitreihe Graz, Mask={mask}°")
    plt.ylabel("DOP")
    plt.show()

    # Anzahl sichtbarer Satelliten
    dop_df["visible_sats"].plot(title=f"Sichtbare Satelliten Graz, Mask={mask}°")
    plt.ylabel("Anzahl Satelliten")
    plt.show()


# --------------------------
# DOP-Berechnung für Narvik
# --------------------------
mask_angles = [0, 5, 10, 15, 20]

for mask in mask_angles:
    dop_df = compute_dop_time_series(
        ecef_values,
        narvik_lat, narvik_lon, narvik_cart,
        mask_angle_deg=mask
    )
    
    # PDOP, HDOP, VDOP Plot
    dop_df[["PDOP", "HDOP", "VDOP"]].plot(
        title=f"DOP-Zeitreihe Narvik, Mask={mask}°"
    )
    plt.ylabel("DOP")
    plt.show()
    
    # Anzahl sichtbarer Satelliten Plot
    dop_df["visible_sats"].plot(
        title=f"Sichtbare Satelliten Narvik, Mask={mask}°"
    )
    plt.ylabel("Anzahl Satelliten")
    plt.show()

"""# --------------------------
# DOP-Berechnung mit ausgeschlossener PRN 4
# --------------------------
dop_df_exclude = compute_dop_time_series(
    ecef_values,
    graz_lat, graz_lon, graz_cart,
    mask_angle_deg=10,
    exclude_prns=[4]
)
dop_df_exclude[["PDOP","HDOP","VDOP"]].plot(title="DOP Graz ohne PRN 4, Mask=10°")
plt.ylabel("DOP")
plt.show()"""