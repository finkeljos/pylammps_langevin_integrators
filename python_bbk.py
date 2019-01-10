#bbk integrator which uses a single normal RV per timestep

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
    """ Python implementation of BBK """

    def __init__(self, ptr):
        super(NVT, self).__init__(ptr)

    def init(self):
        self.units_conversion = self.lmp.extract_global("ftm2v", 1)
        self.ntypes = self.lmp.extract_global("ntypes", 0)
        self.dt = self.lmp.extract_global("dt", 1)
        self.sigma = np.sqrt(2*gamma*kb*T*self.dt*self.units_conversion )
        self.mass = self.lmp.numpy.extract_atom_darray("mass", self.ntypes+1)
        self.b = 1/(1+gamma*self.dt/2)
        nlocal = self.lmp.extract_global("nlocal", 0)

    def initial_integrate(self, vflag):
        nlocal = self.lmp.extract_global("nlocal", 0)
        type = self.lmp.numpy.extract_atom_iarray("type", nlocal)
        x = self.lmp.numpy.extract_atom_darray("x", nlocal, dim=3)
        v = self.lmp.numpy.extract_atom_darray("v", nlocal, dim=3)
        f = self.lmp.numpy.extract_atom_darray("f", nlocal, dim=3)
        mass = self.mass
        a = self.a
        h = self.dt
        sigma = self.sigma
        n = x.shape[0]

        W = np.random.normal(0,1,x.shape)

	# turn mass into array of dimension (2048,) in order to match dimensions of x,v and f
        m = np.take(mass,type)
        mass = np.reshape(m,nlocal,1)

        for i in range(0,3):
            v[:,i] = a*v[:,i] + .5*h*f[:,i]*units/mass + .5*sigma*W[:,i]/np.sqrt(mass)
            x[:,i] = x[:,i] + h*v[:,i]
            v[:,i] = b*v[:,i] + .5*sigma*b*W[:,i]/np.sqrt(mass)

    def final_integrate(self):
        nlocal = self.lmp.extract_global("nlocal", 0)
        type = self.lmp.numpy.extract_atom_iarray("type", nlocal)
        x = self.lmp.numpy.extract_atom_darray("x", nlocal, dim=3)
        v = self.lmp.numpy.extract_atom_darray("v", nlocal, dim=3)
        f = self.lmp.numpy.extract_atom_darray("f", nlocal, dim=3)
        units = self.units_conversion
        mass = self.mass
        b = self.b
        h = self.dt
        sigma = self.sigma
        n = v.shape[0]

	# turn mass into array of dimension (2048,) in order to match dimensions of x,v and f
        m = np.take(mass,type)
        mass = np.reshape(m,nlocal,1)

        for i in range(0,3):
            v[:,i] = v[:,i] + .5*b*h*f[:,i]*units/mass
