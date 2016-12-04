# ece901-project
Repo for ECE901 project

# Setup
* Designed for Python 2.7
* Install virtualenv https://virtualenv.pypa.io/en/stable/ and virtualenvwrapper https://pypi.python.org/pypi/virtualenvwrapper/. Create & load a virtual env, say `ece901_env`:
```sh
mkvirtualenv ece901_env
workon ece901_env
```
* Install/upgrade pip
```sh
pip install --upgrade pip
```
* Install lasagne https://github.com/Lasagne/Lasagne and its dependencies:
```sh
pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt
pip install Lasagne==0.1
```
