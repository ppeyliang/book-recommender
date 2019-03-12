# Book Recommender System

For this project, I built book recommender systems using collaborative filtering and singular value decomposition approach
to help user discover new book that match their need.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The code is written in Python 3.7 using [Jupyter Notebook](https://jupyter.org/install).

You need to install [Surprise](https://github.com/NicolasHug/Surprise/tree/711fb80748140c44e0ed870e573c735307e6c3cc) (a Python library for recommender system) in order to run the code.

[Flask](http://flask.pocoo.org/) is used for development of backend API.

### Installing

With pip (you'll need numpy, and a C compiler. Windows users might prefer using conda):

```
$ pip install numpy
$ pip install scikit-surprise
```

With conda:

```
$ conda install -c conda-forge scikit-surprise
```

## Deployment

I built a backend API and host it on [Pythonanywhere](https://www.pythonanywhere.com/user/ppeyliang/). This is the [website](https://book-recommender.netlify.com/)
for demonstration of user-based collaborative filtering.

The website frontend is developed using [React](https://reactjs.org/) and collaborated with [@ChloeLiang](https://github.com/ChloeLiang) at this [repository](https://github.com/ChloeLiang/book-recommender-demo).
