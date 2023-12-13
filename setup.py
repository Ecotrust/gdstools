from setuptools import setup

setup(
    name='gdstools',
    version='0.1',
    # py_modules=[],
    packages=['gdstools'],
    install_requires=[
        'python-dotenv',
        'earthengine-api>=0.1.35',
    ]
)
