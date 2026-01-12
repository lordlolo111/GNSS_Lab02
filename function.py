import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os

def import_pos_file(filepath):
    df = pd.read_csv(
        filepath,
        comment="#",
        sep=r'\s+',
        header=None,
        names=["system", "satellite", "date", "time", "X", "Y", "Z"]
    )

    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.drop(columns=["time","date"])

    return df

def plot_orbits(ecef, ecsf, satellite_id = None, r=6371000):

    # Kugelparameter
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    # Erde
    x = r * np.outer(np.cos(u),np.sin(v)) # Vektorprodukt
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones_like(u), np.cos(v))

    # falls id einzelner PRN gegeben is
    if satellite_id is not None:
        ecef = ecef[ecef["satellite"] == satellite_id]
        ecsf = ecsf[ecsf["satellite"] == satellite_id]

    # Satellitenbahne [ECEF]
    x_sat1 = ecef["X"].values
    y_sat1 = ecef["Y"].values
    z_sat1 = ecef["Z"].values
    # Satellitenbahnen [ECSF]
    x_sat2 = ecsf["X"].values
    y_sat2 = ecsf["Y"].values
    z_sat2 = ecsf["Z"].values

    # Plotting
    fig = plt.figure(figsize=(14, 8))
    # ECEF
    ax1 = fig.add_subplot(1,2,1, projection="3d")
    ax1.plot_surface(x, y, z, color="lightblue", alpha=0.4,)
    ax1.plot(x_sat1, y_sat1, z_sat1, color="red", linewidth=2, label="Satellite-Orbits")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_zlabel("Z [m]")
    ax1.set_title("GNSS Satellite Orbit in ECEF")
    ax1.set_box_aspect([1, 1, 1])
    #ECSF
    ax2 = fig.add_subplot(1,2,2, projection= "3d")
    ax2.plot_surface(x, y, z, color="lightblue", alpha=0.4,)
    ax2.plot(x_sat2, y_sat2, z_sat2, color="blue", linewidth=2, label="Satellite-Orbits")
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    ax2.set_zlabel("Z [m]")
    ax2.set_title("GNSS Satellite Orbit in ECSF")
    ax2.set_box_aspect([1, 1, 1])

    plt.tight_layout()                   
    plt.show()

def cart_to_geodetic(X,Y,Z):

    # Parameter von WGS-84
    a = 6378137.0 # große Halbachse
    f = 1 / 298.257223563 # Abplattung
    b = a * (1 - f) #kleine Halbachse
    e2  = (a**2 - b**2) / a**2   # erste Exzentrizität^2
    ep2 = (a**2 - b**2) / b**2   # zweite Exzentrizität^2

    p = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Z*a, p*b)

    lon = np.arctan2(Y, X)
    lat = np.arctan2(Z+ep2*b*np.sin(theta)**3, p - e2 * a * np.cos(theta)**3)

    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    h = p/np.cos(lat) - N

    return np.degrees(lat), np.degrees(lon), h

def select_dop_values(ecef, ecsf):
    ecef_values = ecef.set_index("datetime")
    ecsf_values = ecsf.set_index("datetime")
    return ecef_values.between_time("15:00", "20:00"), ecsf_values.between_time("15:00", "20:00")
