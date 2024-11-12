## WARNING: Almost but not fully ready!
To be released with the CHEP2024 proceedings. More information here:
[https://indico.cern.ch/event/1338689/page/31559-the-conference-format](https://indico.cern.ch/event/1338689/page/31559-the-conference-format)

# Development guide
Only tested on machines with Nvidia GPUs.

First, shell inside the container. The `.sif` file can be downloaded from the link below. I use apptainer binary which
can be downloaded from apptainer website. You can just extract it somewhere locally and don't need admin access to
install. I recommend this over CERN/other installed singularity as you get full control and is faster to log in to.


[The container](https://uzh-my.sharepoint.com/:u:/g/personal/shahrukh_qasim_physik_uzh_ch/EbHEeOPLryFFn2N5m7xCHhABl1Hr7KrNrjibF5KB7ctzzw)


And clone the repository (or you can clone my version)
```
git clone https://github.com/jkiesele/fastgraphcompute.git
cd fastgraphcompute
```

And install
```
pip3 install -v --user --no-deps --no-build-isolation .
```
Here, the build isolation mode is important because this way, `pip` will not install dozens of packages every time. The
installation is fairly fast so even while developing, I'd recommend simply installing the package every time you modify
it.

**Important:** go inside `test` directory (don't run from parent directory). Once inside, just run the test case:
```
python3 -m unittest Test_bin_by_coordinates.TestBinByCoordinates
```

Or run one specific test case:
```
python3 -m unittest Test_bin_by_coordinates.TestBinByCoordinates.test_large_scale_cuda
```

Enjoy development!


