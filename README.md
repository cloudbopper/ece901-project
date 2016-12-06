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
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

# Running
main.py is the main script. See cmds.txt for sample command lines. To run the last the command in cmds.txt:
```
tail -n 1 cmds.txt|sh
```
