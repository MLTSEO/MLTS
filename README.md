# MLTS
Machine Learning Toolkit for SEO

Initial demo notebook [here](https://github.com/MLTSEO/MLTS/blob/master/Demos.ipynb)

## What are the problems/needs?
What are the particular problems in the community that could be solved via machine learning.
* Generating better titles.
* Generating descriptions for pages. Summarization.
* Generating alt text from images.
* Need to get from the community.
* Create a Twitterbot

## What is the overall flow?
* Data Getting
* Data Cleaning and Feature Extraction
* Iteration and Updating
* Optimization
* Models (train / predict)

## Roles
* Developing Use Cases
* Evangelism / Community
* Analytics (per Britney: [Analytics](https://ga-beacon.appspot.com/UA-XXXXX-X/gist-id?pixel))
* Coding
* Tutorials
* Documentation / Readability
* Unit Tests / Linting
* Design

## Data needs
* Link data
* Analytics
* Scraping
* Ranking data
* Anonymous performance data

## Proposed Structure
Most folders include a Todo.txt with some suggested items to start with.

* APIs: Holds glue for various SEO APIs
* Data: Holds datagetter classes for APIs and hosted datasets.
* Docs: Holds the documentation for the repo.
* Models: Holds various models that can be used to train on.
* NPL: Glue for NLP libraries
* Testing: Unit testing and CI
* Tutorials: Holds iPython tutorials in Pytorch and Tensorflow
* Config.py: Holds API keys and configuration data.
* Main.py: The main application file.
* requirements.txt: Python libraries needed to install via Pip.


Original concept gist: ([source](https://gist.github.com/jroakes/e84180a6ebafce11cecc9554421a9ac3))
