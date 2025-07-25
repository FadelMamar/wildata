import os
from fractions import Fraction

import pandas as pd
import piexif
from PIL import Image


def decimal_to_dms(decimal_degree):
    """
    Convert decimal degrees to degrees, minutes, seconds tuple for EXIF.
    Returns: (degrees, minutes, seconds)
    """
    abs_value = abs(decimal_degree)
    degrees = int(abs_value)
    minutes_float = (abs_value - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    return degrees, minutes, seconds


class ExifGPSManager:
    """
    Class to manage EXIF GPS data for images, including batch updates from CSV.
    """

    def __init__(self):
        pass

    def _get_gps_data(self, latitude, longitude, altitude=None):
        lat_dms_tuple = decimal_to_dms(abs(latitude))
        lon_dms_tuple = decimal_to_dms(abs(longitude))
        lat_dms = [
            (int(lat_dms_tuple[0]), 1),
            (int(lat_dms_tuple[1]), 1),
            (
                Fraction(lat_dms_tuple[2]).limit_denominator().numerator,
                Fraction(lat_dms_tuple[2]).limit_denominator().denominator,
            ),
        ]
        lon_dms = [
            (int(lon_dms_tuple[0]), 1),
            (int(lon_dms_tuple[1]), 1),
            (
                Fraction(lon_dms_tuple[2]).limit_denominator().numerator,
                Fraction(lon_dms_tuple[2]).limit_denominator().denominator,
            ),
        ]
        lat_ref = "N" if latitude >= 0 else "S"
        lon_ref = "E" if longitude >= 0 else "W"
        gps_data = {
            piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
            piexif.GPSIFD.GPSLatitude: lat_dms,
            piexif.GPSIFD.GPSLatitudeRef: lat_ref,
            piexif.GPSIFD.GPSLongitude: lon_dms,
            piexif.GPSIFD.GPSLongitudeRef: lon_ref,
        }
        if altitude is not None:
            alt_fraction = Fraction(abs(altitude)).limit_denominator()
            gps_data[piexif.GPSIFD.GPSAltitude] = (
                alt_fraction.numerator,
                alt_fraction.denominator,
            )
            gps_data[piexif.GPSIFD.GPSAltitudeRef] = 0 if altitude >= 0 else 1
        return gps_data

    def _get_updated_exif_dict(self, exif_dict, latitude, longitude, altitude=None):
        exif_dict = (
            exif_dict.copy()
            if exif_dict
            else {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        )
        exif_dict["GPS"] = self._get_gps_data(latitude, longitude, altitude)
        return exif_dict

    def add_gps_to_image(
        self, input_path, output_path, latitude, longitude, altitude=None
    ):
        """
        Add GPS coordinates to image EXIF data
        """
        img = Image.open(input_path)
        try:
            exif_dict = piexif.load(img.info.get("exif", b""))
        except:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        exif_dict = self._get_updated_exif_dict(
            exif_dict, latitude, longitude, altitude
        )
        exif_bytes = piexif.dump(exif_dict)
        img.save(output_path, exif=exif_bytes)
        print(f"GPS data added successfully to {output_path}")

    def read_gps_from_image(self, image_path):
        """Read GPS coordinates from image EXIF data"""
        try:
            img = Image.open(image_path)
            exif_dict = piexif.load(img.info.get("exif", b""))
            gps_info = exif_dict.get("GPS", {})
            if not gps_info:
                print("No GPS data found in image")
                return None
            lat_dms = gps_info.get(piexif.GPSIFD.GPSLatitude)
            lat_ref = gps_info.get(piexif.GPSIFD.GPSLatitudeRef, b"N").decode()
            lon_dms = gps_info.get(piexif.GPSIFD.GPSLongitude)
            lon_ref = gps_info.get(piexif.GPSIFD.GPSLongitudeRef, b"E").decode()
            if lat_dms and lon_dms:
                lat_decimal = (
                    lat_dms[0][0] / lat_dms[0][1]
                    + lat_dms[1][0] / (lat_dms[1][1] * 60)
                    + lat_dms[2][0] / (lat_dms[2][1] * 3600)
                )
                if lat_ref == "S":
                    lat_decimal = -lat_decimal
                lon_decimal = (
                    lon_dms[0][0] / lon_dms[0][1]
                    + lon_dms[1][0] / (lon_dms[1][1] * 60)
                    + lon_dms[2][0] / (lon_dms[2][1] * 3600)
                )
                if lon_ref == "W":
                    lon_decimal = -lon_decimal
                print(f"GPS Coordinates: {lat_decimal:.6f}, {lon_decimal:.6f}")
                return lat_decimal, lon_decimal
        except Exception as e:
            print(f"Error reading GPS data: {e}")
            return None

    def add_gps_to_image_inplace(self, file_path, latitude, longitude, altitude=None):
        """
        Add GPS coordinates to image EXIF data (modifies original file)
        """
        try:
            exif_dict = piexif.load(file_path)
        except:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        exif_dict = self._get_updated_exif_dict(
            exif_dict, latitude, longitude, altitude
        )
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, file_path)
        print(f"GPS data added to {file_path}")

    def update_folder_from_csv(
        self,
        image_folder,
        csv_path,
        inplace=False,
        output_dir=None,
        filename_col="filename",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
    ):
        """
        Update EXIF GPS data for all images in a folder using a CSV file.
        CSV must have columns: filename, latitude, longitude, [altitude]
        If inplace is False, images are written to output_dir (preserving filenames).
        """
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            filename = row[filename_col]
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            alt = (
                float(row[alt_col])
                if alt_col in row and not pd.isnull(row[alt_col])
                else None
            )
            img_path = os.path.join(image_folder, filename)
            if inplace:
                self.add_gps_to_image_inplace(img_path, lat, lon, alt)
            else:
                out_dir = output_dir if output_dir is not None else image_folder
                os.makedirs(out_dir, exist_ok=True)
                output_path = os.path.join(out_dir, filename)
                self.add_gps_to_image(img_path, output_path, lat, lon, alt)
