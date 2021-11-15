---
title: Kaggle API Settings
tags: Kaggle Settings Guides
---

## Introduction

Kaggle API are so useful, and can save so much efforts, e.g. access to Kaggle datasets is trivial using the API as demonstrated here below. 

Before one can interact with public Kaggle API, 2 actions should be taken:
- Package Instalation
- Authentication


Next is a detailed description of these 2 steps followed by some API usage examples. 

BTW, when running a notebook in the Kaggle Kernel pllatform, the Kaggle API is available without any the installation and authentication steps.
Yet one more BTW - You should have a Kaggle account for that - sign up in case you are not there already!



## Kaggle API Package Installation

Simply use pip (package installaerfor python):

```python
pip install kaggle
```


## Authentication

To authenticate, one should generate a token json file and store it under  ~/.kaggle (in linux/osx) or under C:\Users<Windows-username>.kaggle\ in windows. So do as follows:


### Token Generation

1. Login to your Kaggle Account
2. Click on user profile picture (upper right) -> Account -> Create New API Token
3. The kaggle.json token file should be downloaded to your local storage. 
4. Create a directory:  
  ```python
  mkdir ~/.kaggle
  ```
5. Copy token file to the created directory:
   ```python
  cp ~/Downloads/kaggle.json  ~/.kaggle
  ```
 
Now Kaggle API is ready for use.



## Kaggle API Usage Examples

(In the below comands the extrac exclamation sign prefix is needed to run commands from within a jupyter notebook)/

- List Kaggle datasets:

```python
!kaggle datasets list
```

- Doenload a dataset:

```python
!kaggle datasets download -d coloradokb/dandelionimages
```



That's it. Enjoy!













