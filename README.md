# TOMOMAK

TOMOMAK is an easy-to-use cross-platform framework for multidimensional limited data tomography. 
The main features of the TOMOMAK framework are:
* Arbitrary number of dimensions.
* Support of different coordinate systems.
* Limited data treating: regularization or model restrictions.
* Possibility to combine different algorithms.
* Arbitrary detector geometry support.
* Algorithms for the synthetic data generation.
* Calculation of the reconstruction quality criteria.
* Hardware acceleration using GPU.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
Usage of [Anaconda](https://https://www.anaconda.com/) is recommended.
Use the following command in order to install required packages:
```
pip install <package name>
```
or if you use Anaconda:
```
conda install <package name>
```
Required packages:

* Shapely (for 2D geometry).
* CuPy (for GPU acceleration).
* Mayavi (for 3D visualization).

If you don't use Anaconda:
* SciPy
### Installing

Copy repository to your computer.

```
---
```
##### Quick Start
The best way to understand the framework  is to use it. Start with the  [basic functionality example.](https://github.com/Koliaska/tomomak/blob/master/examples/basic%20functionality.py)

As soon as you are done switch to the other examples and documentation.



## Documentation
To generate API-doumentation in your terminal or Anaconda Prompt, go to doc folder:
```
cd doc
```
Use pdoc:
```
pdoc --html --force ../tomomak
```

## Authors

* **Nikolai Bakharev** - *PhD, Researcher at Ioffe Institute, St.-Petersburg, Russian Federation* 


## License

This project is licensed under the Revised BSD License - see the [LICENSE.txt](LICENSE.txt) file for details

## Contacts
If you have any questions, proposals or you simply don't know, is it possible to use this framework in your study - don't hesitate to contact [**the author.**](https://github.com/Koliaska/) 

