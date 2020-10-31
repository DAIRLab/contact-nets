from setuptools import setup

setup(
    name="lemkelcp",
    version='0.1',
    author="Andy Lamperski",
    author_email="alampers@umn.edu",
    description=("A Python implementation of Lemke's Algorithm for linear complementarity problems"),
    license="MIT",
    url="https://github.com/AndyLamperski/lemkelcp",
    install_requires=['numpy'],
    keywords="linear complementarity problem lcp optimization",
    packages=['lemkelcp'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics"
    ]
    
)
