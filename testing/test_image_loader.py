import unittest
import ee
from gdstools import GEEImageLoader

class TestGEEImageLoader(unittest.TestCase):
    def setUp(self):
        ee.Initialize()
        self.image_id = 'LANDSAT/LC08/C01/T1_TOA/LC08_044034_20140318'
        image = ee.Image(self.image_id)
        self.loader = GEEImageLoader(image)

    def test_get_property(self):
        self.assertEqual(self.loader.get_property('system:id'), self.image_id)

    def test_set_property(self):
        self.loader.set_property('system:time_start', 1395120000000)
        self.assertEqual(self.loader.get_property('system:time_start'), 1395120000000)

    def test_get_params(self):
        self.assertEqual(self.loader.get_params('crs'), 'EPSG:4326')

    def test_set_params(self):
        self.loader.set_params('region', ee.Geometry.Point([-122.262, 37.8719]))
        self.assertEqual(self.loader.get_params('region').getInfo(), {'type': 'Point', 'coordinates': [-122.262, 37.8719]})

    def test_get_viz_params(self):
        self.assertEqual(self.loader.get_viz_params('bands'), ['B4', 'B3', 'B2'])

    def test_set_viz_params(self):
        self.loader.set_viz_params('bands', ['B5', 'B4', 'B3'])
        self.assertEqual(self.loader.get_viz_params('bands'), ['B5', 'B4', 'B3'])

    def test_get_url(self):
        url = self.loader.get_url()
        self.assertTrue(url.startswith('https://earthengine.googleapis.com/v1alpha/projects/earthengine-legacy/thumbnails'))

    def test_metadata_from_collection(self):
        collection_id = 'LANDSAT/LC08/C01/T1_TOA'
        collection = ee.ImageCollection(collection_id).filterBounds(ee.Geometry.Point([-122.262, 37.8719])).limit(10)
        metadata = self.loader.metadata_from_collection(collection)
        self.assertEqual(metadata['id'], collection_id)
        self.assertEqual(metadata['provider'], 'USGS')
        self.assertEqual(metadata['size'], 10)
        self.assertEqual(metadata['bands'], ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'sr_aerosol', 'pixel_qa'])

        
