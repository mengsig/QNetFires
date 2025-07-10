# Installation

If you are using **arch** linux, you can install directly via the command:

```
bash install.sh
```

Otherwise, on any **OS** you can install by first ensuring that you are using
python version **~Python3.10**. Thereafter, execute the following commands:

```
git clone git@github.com:pyregence/pyretechnics.git
git clone git@github.com:mengsig/DomiRank.git
python -m venv qnetfires
source qnetfires/bin/activate
pip install -e DomiRank/.
pip install -r requirements.txt
cd pyregence
python setup.py install
cd ..
```

# Running Simulations and Creating Fire Breaks
The ```run.sh``` script creates an easy script that will run all of 
the required files in the correct order, by passing consistent argument
types. Moreover, it does this for a list of specified centralities.
To change the size of the lanscape, please navigate to run.sh and change
the variables ```XLEN``` and ```YLEN``` to your desired size. Thereafter,
a savename should be given that will store all results in *src/results/<savename>*.
Once this is done, simply run:

```
bash run.sh
```

And all information should be stored in the *src/results/<savename>/* folder.

Enjoy (:
