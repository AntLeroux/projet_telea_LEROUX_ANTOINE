# -*- coding: utf-8 -*-
"""
@author: Antoine Leroux

Bibliothèque spécifique au projet de télédétéction avancée.
Contient les fonctions de pré-traitement, de stacking et de visualisation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, ogr

gdal.UseExceptions()

def rasterization(in_vector, ref_image, field_name):
    """
    Rasterise un shapefile en mémoire pour correspondre au gabarit de ref_image.
    """
    # Récupération de la géométrie de référence
    ref_ds = gdal.Open(ref_image, gdal.GA_ReadOnly)
    geo_transform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    cols = ref_ds.RasterXSize
    rows = ref_ds.RasterYSize
    
    # Ouverture du vecteur
    driver_ogr = ogr.GetDriverByName("ESRI Shapefile")
    source_ds = driver_ogr.Open(in_vector, 0)
    if not source_ds:
        raise FileNotFoundError(f"Impossible d'ouvrir le vecteur : {in_vector}")
    source_layer = source_ds.GetLayer()
    
    # Création du raster temporaire en mémoire (MEM)
    target_ds = gdal.GetDriverByName('MEM').Create('', cols, rows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    
    # Initialisation à 0
    band = target_ds.GetRasterBand(1)
    band.Fill(0)
    band.SetNoDataValue(0)
    
    # Rasterisation (Le vecteur est "imprimé" sur le raster)
    gdal.RasterizeLayer(target_ds, [1], source_layer, options=[f"ATTRIBUTE={field_name}"])
    
    # Lecture en array numpy
    mask_array = band.ReadAsArray()
    
    # Nettoyage
    target_ds = None
    ref_ds = None
    source_ds = None
    
    return mask_array


def calculate_ari(path_b03, path_b05, output_path, dates, nodata_val=-9999):
    """
    Calcule l'indice ARI (B05 - B03) / (B05 + B03) et sauvegarde le résultat.
    """
    print(f"Calcul ARI -> {os.path.basename(output_path)}")
    
    ds_b03 = gdal.Open(path_b03)
    ds_b05 = gdal.Open(path_b05)

    cols = ds_b03.RasterXSize
    rows = ds_b03.RasterYSize
    bands_count = ds_b03.RasterCount
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, cols, rows, bands_count, gdal.GDT_Float32)
    out_ds.SetGeoTransform(ds_b03.GetGeoTransform())
    out_ds.SetProjection(ds_b03.GetProjection())

    for b in range(1, bands_count + 1):
        # Lecture
        arr_b03 = ds_b03.GetRasterBand(b).ReadAsArray().astype(np.float32)
        # Lecture B05 avec redimensionnement immédiat (20m -> 10m)
        arr_b05 = ds_b05.GetRasterBand(b).ReadAsArray(
            0, 0, ds_b05.RasterXSize, ds_b05.RasterYSize,
            buf_xsize=cols, buf_ysize=rows
        ).astype(np.float32)

        # Calcul sécurisé (éviter division par 0)
        ari = np.full(arr_b03.shape, nodata_val, dtype=np.float32)
        denom = arr_b05 + arr_b03
        
        valid = (denom != 0) & (arr_b03 != 0) & (arr_b05 != 0)
        ari[valid] = (arr_b05[valid] - arr_b03[valid]) / denom[valid]

        # Écriture
        out_band = out_ds.GetRasterBand(b)
        out_band.WriteArray(ari)
        out_band.SetNoDataValue(nodata_val)
        
        if dates and (b-1 < len(dates)):
            out_band.SetDescription(dates[b-1])

    out_ds = None
    ds_b03 = None
    ds_b05 = None


def stack(paths_dict, ari_path, dates, output_path):
    """
    Fusionne les bandes spectrales et l'ARI en un seul fichier GeoTIFF.
    Renommée 'build_stack' pour correspondre au notebook.
    """
    
    # Image de référence (B02 pour la géométrie 10m)
    ref_ds = gdal.Open(paths_dict['B02'])
    cols, rows = ref_ds.RasterXSize, ref_ds.RasterYSize
    geo, proj = ref_ds.GetGeoTransform(), ref_ds.GetProjection()
    
    # Calcul du nombre total de bandes
    nb_dates = len(dates)
    has_ari = os.path.exists(ari_path)
    
    # Nb types de bandes * Nb Dates + (ARI * Nb Dates si présent)
    total_bands_out = (len(paths_dict) * nb_dates)
    if has_ari:
        ds_ari = gdal.Open(ari_path)
        total_bands_out += ds_ari.RasterCount
        
    # Création fichier sortie
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, cols, rows, total_bands_out, gdal.GDT_Float32)
    out_ds.SetGeoTransform(geo)
    out_ds.SetProjection(proj)
    
    current_band_idx = 1

    # Ajout Bandes Spectrales
    for band_name, file_path in paths_dict.items():
        ds_in = gdal.Open(file_path)
        if not ds_in: continue
            
        for t in range(1, ds_in.RasterCount + 1):
            data = ds_in.GetRasterBand(t).ReadAsArray(
                0, 0, ds_in.RasterXSize, ds_in.RasterYSize,
                buf_xsize=cols, buf_ysize=rows
            ).astype(np.float32)
            
            out_band = out_ds.GetRasterBand(current_band_idx)
            out_band.WriteArray(data)
            out_band.SetNoDataValue(0)
            
            d_name = dates[t-1] if (dates and t-1 < len(dates)) else f"T{t}"
            out_band.SetDescription(f"{band_name}_{d_name}")
            
            current_band_idx += 1

    # Ajout ARI
    if has_ari:
        for t in range(1, ds_ari.RasterCount + 1):
            data = ds_ari.GetRasterBand(t).ReadAsArray().astype(np.float32)
            out_band = out_ds.GetRasterBand(current_band_idx)
            out_band.WriteArray(data)
            
            d_name = dates[t-1] if (dates and t-1 < len(dates)) else f"T{t}"
            out_band.SetDescription(f"ARI_{d_name}")
            current_band_idx += 1

    out_ds = None


def plot_phenology(ari_path, mask_array, output_fig_path, dates):
    """
    Génère et sauvegarde le graphique de phénologie sans l'afficher (plt.show() retiré).
    C'est le notebook qui gérera l'affichage.
    """
    
    ari_ds = gdal.Open(ari_path)
    ari_stack = ari_ds.ReadAsArray() # (Temps, Rows, Cols)

    ids_classes = [1, 2, 3, 4]
    strate_names = {1: "Sol Nu", 2: "Herbe", 3: "Landes", 4: "Arbre"}
    strate_colors = {1: 'peru', 2: 'khaki', 3: 'lightgreen', 4: 'lightseagreen'}
    
    plt.figure(figsize=(10, 6))
    x_axis = range(len(dates))

    for c in ids_classes:
        rows_idx, cols_idx = np.where(mask_array == c)
        
        if len(rows_idx) > 0:
            values_time = ari_stack[:, rows_idx, cols_idx]
            masked_values = np.ma.masked_equal(values_time, -9999)
            
            means = np.mean(masked_values, axis=1)
            stds = np.std(masked_values, axis=1)
            
            plt.plot(x_axis, means, label=strate_names[c], color=strate_colors[c], linewidth=2, marker='o')
            plt.fill_between(x_axis, means - stds, means + stds, color=strate_colors[c], alpha=0.2)

    plt.title("Phénologie des strates (Indice ARI)", fontsize=14, fontweight='bold')
    plt.xlabel("Dates", fontsize=12)
    plt.ylabel("Valeur ARI moyenne", fontsize=12)
    plt.xticks(x_axis, dates, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_fig_path)