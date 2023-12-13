import unittest
from pyproj import CRS
from gdstools.utils import infer_utm

class TestInferUTM(unittest.TestCase):
    def test_infer_utm(self):
        bbox = [-118.21743236065831, 44.65668546656683, -118.2169795936661, 44.657008631159506]
        expected_crs = CRS.from_epsg(32611)
        self.assertEqual(infer_utm(bbox), expected_crs)

if __name__ == '__main__':
    unittest.main()
