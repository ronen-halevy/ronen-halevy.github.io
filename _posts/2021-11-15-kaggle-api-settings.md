---
title: Kaggle API Settings
tags: Kaggle Settings Guides
---

## Introduction


This post describes the setup steps required before using `Kaggle API`. These settins are required to execute Kaggle API outside the `Kaggle Kernel pllatform`, e.g. in `Colab` or in a private computing platform. When executing a notebook inside Kaggle Kernel platform, the described settings are not needed. Issue is presented [here](/guides/content/editing-an-existing-page).


`Kaggle API` can be so useful, and save much efforts that would otherwise be required, e.g. consuming Kaggle datasets is trivial with the API, but would have required a pre-arrangement of private storage for the dataset - in a local pc or in the cloud, according to executiion mode - otherwise. 

Before one can interact with public Kaggle API, 2 actions should be taken:
- Package Instalation
- Authentication


Next is a detailed description of these 2 steps followed by some API usage examples. 

BTW - You should have a Kaggle account for that - sign up in case you are not there already!


## Kaggle API Package Installation

Simply use pip (package installaerfor python):

```python
pip install kaggle
```


## Authentication

To authenticate, one should do 2 actions:
1. Generate a token file - as detailed next.
2. Store the token file under
```python
~/.kaggle
```
in linux/osx or under
```python
C:\Users<Windows-username>.kaggle\ 
```

Here are details instructions for doing that:

### Generate a token file

1. Login to your Kaggle Account
2. In Kaggle web page click: `user profile picture (upper right of page)` -> `Account` -> `Create New API Token`
3. The kaggle.json token file should be downloaded to your local storage. 


### Store the token


1. Create a directory:  
  ```python
  mkdir ~/.kaggle
  ```
2. Copy token file to the created directory:
   ```python
  cp ~/Downloads/kaggle.json  ~/.kaggle
  ```
Beware! the token file is now exposed, so you might want to change it's permissions:

```python
chmod 600 ~/.kaggle/kaggle.json
```

***Notes on setting the token in Colab***


If running in Colab, then the token file should be uploaded to the platform. Here below is a section of code that should be addd to the Colab notebook. This section provides an iteractive file upload button for uploading `kaggle.json`.

```python
from google.colab import files

uploaded = files.upload()
  
# Move kaggle.json into ~/.kaggle:
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```


Now Kaggle API is ready to go.



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













