---
title: Kaggle API Settings
tags: Kaggle Settings Guides
---

## Introduction


This post describes the setup steps required before using Kaggle API. These settins are needed when Kaggle API is used outside Kaggle Kernel pllatform, e.g. in Colab or from a private platform. When in Kaggle Kernel pllatform, Kaggle API is ready to use without  these settings. 

Kaggle API can be so useful, and save much efforts that would otherwise be required , e.g. when using Kaggle datasets - it is trivial by using the API but would have required a private storage for the used dataset otherwise. 

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

To authenticate, one should generate a token json file - as detailed next -  and store it under
```python
~/.kaggle
```
in linux/osx or under
```python
C:\Users<Windows-username>.kaggle\ 
```

So do as follows:

1. Login to your Kaggle Account
2. In Kaggle web page click: user profile picture (upper right) -> Account -> Create New API Token
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

(In the below comands the extrac exclamation sign prefix is needed to run commands from within a jupyter notebook).

- List Kaggle datasets:

```python
!kaggle datasets list
```

- Doenload a dataset:

```python
!kaggle datasets download -d coloradokb/dandelionimages
```



That's it. Enjoy!













