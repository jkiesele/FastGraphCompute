
#include "binstepper.h"

//create test instance of binstepper
int nbins[3] = {3,3,3};
int binoffset[3] = {0,0,0};

binstepper<3> binst(nbins,binoffset);