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
        h = self.lmp.extract_global("dt", 1)
        ftm2v = self.lmp.extract_global("ftm2v", 1)
        self.ntypes = self.lmp.extract_global("ntypes", 0)
        self.dt = h
        self.sigma = np.sqrt(2*gamma*kb*T*h*ftm2v)
        self.mass = self.lmp.numpy.extract_atom_darray("mass", self.ntypes+1)
        self.a = (1-gamma*h/2)/(1+gamma*h/2)
        self.b = 1/(1+gamma*h/2)

    def initial_integrate(self, vflag):
        nlocal = self.lmp.extract_global("nlocal", 0)
        type = self.lmp.numpy.extract_atom_iarray("type", nlocal)
        x = self.lmp.numpy.extract_atom_darray("x", nlocal, dim=3)
        v = self.lmp.numpy.extract_atom_darray("v", nlocal, dim=3)
        f = self.lmp.numpy.extract_atom_darray("f", nlocal, dim=3)
        ftm2v = self.lmp.extract_global("ftm2v", 1)
        mass = self.mass
        a = self.a
        b = self.b
        h = self.dt
        sigma = self.sigma
        Z = np.random.normal(0,1,x.shape)
        n  = x.shape[0]

        # turn mass into array of dimension (2048,) in order to match x,v,f dimensions
        m=np.take(mass,type)
        mass=np.reshape(m,nlocal,1)

        for i in range(0,3):  #loop over 3 dimensions
            x[:,i] = x[:,i] + b*v[:,i]*h + .5*b*h**2*f[:,i]*ftm2v/mass + .5*b*h*sigma*Z[:,i]/np.sqrt(mass)
            v[:,i] = a*v[:,i] + (.5*a*ftm2v*f[:,i]*h)/mass + b*sigma*Z[:,i]/np.sqrt(mass)

    def final_integrate(self):
        nlocal = self.lmp.extract_global("nlocal", 0)
        mass = self.lmp.numpy.extract_atom_darray("mass", self.ntypes+1)
        type = self.lmp.numpy.extract_atom_iarray("type", nlocal)
        x = self.lmp.numpy.extract_atom_darray("x", nlocal, dim=3)
        v = self.lmp.numpy.extract_atom_darray("v", nlocal, dim=3)
        f = self.lmp.numpy.extract_atom_darray("f", nlocal, dim=3)
        ftm2v = self.lmp.extract_global("ftm2v", 1)
        n = v.shape[0]
        h = self.dt
        sigma = self.sigma
        mass = self.mass

	# turn mass into array of dimension (2048,) in order to match dimensions with x,v,f
        m=np.take(mass,type)
        mass=np.reshape(m,nlocal,1)

        for i in range(0,3):  #loop over # of dimensions
            v[:,i] += .5*ftm2v*f[:,i]*h/mass

