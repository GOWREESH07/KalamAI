"""
Unit tests for spectral index formulas and AOI area calculation.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from backend.pipeline.features import compute_ndvi, compute_ndwi, compute_lswi
from backend.pipeline.aoi import compute_area_km2, validate_aoi


class TestNDVI:
    def test_positive_vegetation(self):
        nir = np.array([0.5])
        red = np.array([0.1])
        result = compute_ndvi(nir, red)
        assert abs(result[0] - (0.5 - 0.1) / (0.5 + 0.1)) < 1e-6

    def test_water_negative(self):
        nir = np.array([0.05])
        red = np.array([0.15])
        result = compute_ndvi(nir, red)
        assert result[0] < 0

    def test_zero_denominator(self):
        nir = np.array([0.0])
        red = np.array([0.0])
        result = compute_ndvi(nir, red)
        assert np.isnan(result[0])

    def test_range_clamped(self):
        nir = np.random.rand(100)
        red = np.random.rand(100)
        result = compute_ndvi(nir, red)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1.0) and np.all(valid <= 1.0)


class TestNDWI:
    def test_water_positive(self):
        green = np.array([0.3])
        nir = np.array([0.05])
        result = compute_ndwi(green, nir)
        assert result[0] > 0

    def test_land_negative(self):
        green = np.array([0.1])
        nir = np.array([0.4])
        result = compute_ndwi(green, nir)
        assert result[0] < 0


class TestLSWI:
    def test_moist_positive(self):
        nir = np.array([0.5])
        swir = np.array([0.2])
        result = compute_lswi(nir, swir)
        assert result[0] > 0


class TestAOI:
    def test_area_within_bounds(self):
        cfg = {
            "aoi": {
                "lon_min": 77.47, "lon_max": 77.53,
                "lat_min": 8.680, "lat_max": 8.725,
                "epsg": 32643, "min_area_km2": 20.0
            }
        }
        area = compute_area_km2(cfg)
        # Approximate expected: ~33 km²
        assert 25 < area < 50, f"Area {area} outside expected range"

    def test_validate_passes(self):
        cfg = {
            "aoi": {
                "lon_min": 77.47, "lon_max": 77.53,
                "lat_min": 8.680, "lat_max": 8.725,
                "epsg": 32643, "min_area_km2": 20.0
            }
        }
        area = validate_aoi(cfg)
        assert area >= 20.0

    def test_validate_fails_small_aoi(self):
        cfg = {
            "aoi": {
                "lon_min": 77.47, "lon_max": 77.472,
                "lat_min": 8.680, "lat_max": 8.682,
                "epsg": 32643, "min_area_km2": 20.0
            }
        }
        with pytest.raises(ValueError):
            validate_aoi(cfg)
