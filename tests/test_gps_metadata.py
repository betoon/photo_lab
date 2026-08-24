from imaging import _gps_ratio_to_float, _gps_values_to_degrees, _parse_gps_info


class ExifReadRatio:
    def __init__(self, num, den=1):
        self.num = num
        self.den = den


def test_exifread_ratio_and_gps_conversion():
    values = [ExifReadRatio(40), ExifReadRatio(30), ExifReadRatio(0)]
    assert _gps_ratio_to_float(ExifReadRatio(3, 2)) == 1.5
    assert _gps_values_to_degrees(values, "N") == 40.5
    assert _gps_values_to_degrees(values, "S") == -40.5


def test_pillow_gps_dictionary_conversion():
    # Pillow's numeric GPS keys: latitude ref/value, longitude ref/value.
    gps = {1: "N", 2: (40, 30, 0), 3: "W", 4: (74, 0, 0)}
    assert _parse_gps_info(gps) == (40.5, -74.0)
