from __future__ import print_function
import lammps
import ctypes
import traceback
import numpy as np


#global parameters
kb=0.0019872041
gamma=1.0
T=450.0

#generate seed. this is different than how RVs generated for 1st set of runs
r=np.random.randint(1,1e6)
np.random.seed(r)


class LAMMPSIntegrator(object):
    def __init__(self, ptr):
        self.lmp = lammps.lammps(ptr=ptr)

    def init(self):
        pass

    def initial_integrate(self, vflag):
        pass

    def final_integrate(self):
        pass

    def initial_integrate_respa(self, vflag, ilevel, iloop):
        pass

    def final_integrate_respa(self, ilevel, iloop):
        pass

    def reset_dt(self):
        pass

class NVT(LAMMPSIntegrator):
    """ Python implementation of fix/nve """
    def __init__(self, ptr):
        super(NVT, self).__init__(ptr)

    def init(self):
        self.h = self.lmp.extract_global("dt", 1)
        self.units_conversion = self.lmp.extract_global("ftm2v", 1)
        self.ntypes = self.lmp.extract_global("ntypes", 0)
        self.sigma = np.sqrt(2*gamma*kb*T*self.h*self.units_conversion)
        self.c1 = np.exp(-gamma*self.h)
        self.c2 = np.sqrt(self.units_conversion*kb*T*(1-np.exp(-2*gamma*self.h)))

    def initial_integrate(self, vflag):
        nlocal = self.lmp.extract_global("nlocal", 0)
        type = self.lmp.numpy.extract_atom_iarray("type", nlocal)
        x = self.lmp.numpy.extract_atom_darray("x", nlocal, dim=3)
        v = self.lmp.numpy.extract_atom_darray("v", nlocal, dim=3)
        f = self.lmp.numpy.extract_atom_darray("f", nlocal, dim=3)
        mass = self.lmp.numpy.extract_atom_darray("mass", self.ntypes+1)
        h = self.h
        n = x.shape[0]
        c1=self.c1
        c2=self.c2
        Z = np.random.normal(0,1,x.shape)

        # turn mass into array of dimension (2048,) in order to match x,v,f
        m = np.take(mass,type)
        mass = np.reshape( m, nlocal, 1)

        for i in range(0,3):   #loop over # of dimensions
            v[:,i] = v[:,i] + .5*h*ftm2v*f[:,i]/mass
            x[:,i] = x[:,i] + .5*h*v[:,i]
            V = c1*v[:,i] + c2*Z[:,i]/np.sqrt(mass[:])
            x[:,i] = x[:,i] + .5*h*V
            v[:,i] = V

    def final_integrate(self):
        nlocal = self.lmp.extract_global("nlocal", 0)
        mass = self.lmp.numpy.extract_atom_darray("mass", self.ntypes+1)
        type = self.lmp.numpy.extract_atom_iarray("type", nlocal)
        x = self.lmp.numpy.extract_atom_darray("x", nlocal, dim=3)
        v = self.lmp.numpy.extract_atom_darray("v", nlocal, dim=3)
        f = self.lmp.numpy.extract_atom_darray("f", nlocal, dim=3)
        n = x.shape[0]
        h = self.h

        # turn mass into array of dimension (2048,) in order to match x,v,f
        m = np.take(mass,type)
        mass = np.reshape( m, nlocal, 1)

        for i in range(0,3):   #loop over # of dimensions
            v[:,i] += .5*h*f[:,i]*self.units_conversion/mass

