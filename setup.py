from setuptools import setup

setup(
    name='gdstools',
    py_modules=[],
    install_requires=[
        'numpy',
        'matplotlib',
        'tqdm',
        'python-dotenv',
        'earthengine-api>=0.1.35',
        'geetools',
        'rasterio',
        'pandas',
        'seaborn'
    ]
)
