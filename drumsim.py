import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import numpy as np
import pandas as pd
from scipy import special as bes
from scipy.integrate import quad
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import os
from mpl_toolkits.mplot3d import Axes3D 
from IPython.display import HTML
from moviepy.editor import *

class Mode: 
    '''
    Class representing a single vibrational mode that can form on a circular drum of certain
    radius and wave speed. 
    
    A single mode is parametrised by integers m (>=0) and n (>=1) denoting the 
    number of diametral and circular nodes respectively
    '''
    
    def __init__(self, m, n, radius, c, u0=None):
        
        if(m<0 or n<=0 or not isinstance(m+n, int)):
            print(m)
            print(n)
            raise ValueError('m and n have to be non-negative and positive integers respectively')
            
        self.m = m
        self.n = n
        self.radius = radius
        self.c = c
        
        # Define lambda value for specific mode
        self.lam = bes.jn_zeros(m,n)[n-1]/self.radius
        
        self.freq = self.getFrequency()
        self.u0 = u0

        # Amplitudes for general solution, provided initial condition u0
        self.a_mn = self.getA()
        self.b_mn = self.getB()
        
        # Displacement 
        self.u = self.getDisp
        
        # Animated frames for one period
        self.frames = None
    
    def getDisp(self, r, theta, t):
        '''
        Returns displacement of membrane at provided radial position, angle and time;
        characterises evolution of drum surface over space and time for this mode
        
        Parameters:
        r: radial position
        theta: anguar position
        t: time 
        
        Returns:
        displacement of membrane
        '''
        m = self.m
        n = self.n
        
        # Decay rate conditional to whether initial conditions have been provided
        decay = np.exp(-0.5*t)
        
#         if(self.u0 == None):
#             decay = 1
            
#         THETA = self.a_mn*np.cos(m*theta)+self.b_mn*np.sin(m*theta)
#         TIME = (np.cos(self.c*self.lam*t)+np.sin(self.c*self.lam*t))*decay
#         R = bes.jv(m, self.lam*r)

#         return R*THETA*TIME

        R = bes.jv(m, self.lam*r)
        THETA = self.a_mn*np.cos(m*theta)+self.b_mn*np.sin(m*theta)
        TIME = np.cos(self.c*self.lam*t)*decay

        if(self.u0 == None):
            decay = 1    
            TIME = (np.cos(self.c*self.lam*t)+np.sin(self.c*self.lam*t))*decay
            
        return R*THETA*TIME


    def sim(self, fpath):
        '''
        Simulates the nm-th mode for one period and saves to path 
    
        Parameters:
        path: where simulation is to be saved
        '''
        fname = self.toString()
        fps = 30 # frame per sec
        
        # See if animation had been previously run to save time
        if(self.frames==None):    
            m = self.m
            n = self.n
            
            N = 150 # Meshsize
            frn = 50 # frame number of the animation

            # Calculate displacements
            r = np.linspace(0, self.radius, N)
            p = np.linspace(0, 2*np.pi, N)
            R, P = np.meshgrid(r, p)
            zarray = np.zeros((N, N, frn))
            X, Y = R*np.cos(P), R*np.sin(P)

            period = 1/self.freq

            for i in range(frn):
                zarray[:,:,i] = self.u(R,P,i*period/frn)

            # Create plot
            amp = np.amax(zarray[:,:,0])
            fig = plt.figure(figsize=(15,10))
            ax = fig.add_subplot(111, projection='3d')
            ax.title.set_text('The Vibration of {}'.format(self.toString()))
            ax.set_xlabel('x',fontsize=20)
            ax.set_ylabel('y', fontsize=20)
            ax.set_zlabel('u', fontsize=20)
            
            
            # Scale z-axis according to intensity of mode
            ax.set_zlim(-(m+2)*amp,(m+2)*amp)

            #Animate 
            
            initial = ax.plot_surface(X,Y,zarray[:,:,0], cmap=plt.cm.magma, antialiased=False, 
                            rcount=14, ccount=14,
                            vmin=-1.5*amp, vmax=1.5*amp, linewidth=0.005, edgecolor='black')
            plot = [initial];

            # Save initial frame to path
            fname = self.toString()

            if not os.path.exists(fpath):
                os.makedirs(fpath)

            path = os.path.join(fpath, fname)
            fig.savefig(path+".png")
            
            def Animate(frn, zarray, plot):
                plot[0].remove()
                surf = ax.plot_surface(X, Y, zarray[:,:,frn],  cmap=plt.cm.magma, antialiased=False, 
                                rcount=int(10+4*m), ccount=int(10+4*n),
                                vmin=-1.5*amp, vmax=1.5*amp, linewidth=0.005, edgecolor='black')
                plot[0] = surf

            ani = animation.FuncAnimation(fig, Animate, frn, fargs=(zarray, plot), interval=1000/fps)
            self.frames = ani
        
        # Save animated gif to path
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        
        fname = self.toString()
        path = os.path.join(fpath, fname)
        self.frames.save(path+".mp4", fps=fps)
        
        clip = (VideoFileClip(path+'.mp4'))
        clip.write_gif(path+'.gif',)
        os.remove(path+'.mp4')
        
        
    def getA(self):
        '''
        Returns:
        Amplitude of the cosine term in the general solution (default=1)
        '''
        if(self.u0== None): return 1
        
        # Calculate double integral
        integrand1 = lambda r,theta: self.u0(r=r,theta=theta)*bes.jv(self.m, self.lam*r)*np.cos(self.m*theta)*r
        integrand2 = lambda theta: quad(integrand1, 0, self.radius, args=(theta))[0]
        double = quad(integrand2,0,2*np.pi)[0]
        
        # Obtain coefficient based on value of "m"
        denom = np.pi*(self.radius*bes.jv(self.m+1, self.lam*self.radius))**2
        if(self.m==0):
            coef = 1/denom
        else:
            coef = 2/denom
        
        return coef*double
    
    def getB(self):
        '''
        Returns:
        Amplitude of the sine term in the general solution (default=1)
        '''
        if(self.u0== None): return 1
        if(self.m==0): return 0
        
        # Calculate double integral
        integrand1 = lambda r,theta: self.u0(r=r,theta=theta)*bes.jv(self.m, self.lam*r)*np.sin(self.m*theta)*r
        integrand2 = lambda theta: quad(integrand1, 0, self.radius, args=(theta))[0]
        double = quad(integrand2,0,2*np.pi)[0]
        
        # Obtain coefficient
        coef = 2/(np.pi*(self.radius*bes.jv(self.m+1, self.lam*self.radius))**2)
        
        return coef*double
    
    def getFrequency(self):
        '''
        Returns:
        Characteristic frequency of the mode
        '''
        return (self.lam*self.c)/(2*np.pi)
    
    
    def getAmp(self):
        '''
        Returns:
        Amplitude of the mode
        '''
        return np.sqrt(self.a_mn**2 + self.b_mn**2)
    
    def toString(self):
        '''
        Returns:
        String representation of mode in format "(m,n)"
        '''
        return '({},{})'.format(self.m,self.n)
        

class CircularDrum:
    '''
    Class representing a circular drum membrane and the general vibrations that form for any initial
    pertubration u0 on its surface.
    The accuracy of the general vibration is bounded above by the superposition of all normal modes 
    leading up to (maxm, maxn)
    '''
    
    def __init__(self, radius, c, u0, maxm, maxn):
        self.u0 = u0
        self.radius = radius
        self.c = c
        
        # All normal modes (up to (maxm, maxn)) that will be superposed to yield general vibration
        self.modes = [Mode(m, n, radius, c, u0) for m in range(0, maxm+1) for n in range(1,maxn+1)]
        
        # Frames of simulation animation
        self.frames = None
    
    def sim(self, fname, fpath, t=5):
        '''
        Simulates drum surface given initial perturbation (general solution) for duration t
        
        Parameters:
        fname: name of file to be saved
        fpath: where file is to be saved
        t: duration of animation (s) 
        
        '''
        
        fps = 30 # frames per sec
        
        if(self.frames==None or t!=5):
            N = 50 # Meshsize
            frn = t*fps # frame number of the animation

            # Calculate displacements
            r = np.linspace(0, self.radius, N)
            p = np.linspace(0, 2*np.pi, N)
            R, P = np.meshgrid(r, p)
            zarray = np.zeros((N, N, frn))
            X, Y = R*np.cos(P), R*np.sin(P)

            period = 1/self.modes[0].freq

            for i in range(frn):
                totaldisp = sum([mode.u(R,P,i*period/(fps*2)) for mode in self.modes])

                zarray[:,:,i] = totaldisp

            amp = np.amax(zarray[:,:,0])
            fig = plt.figure(figsize=(15,10))
            ax = fig.add_subplot(111, projection='3d')
            ax.title.set_text(fname)
            ax.set_xlabel('x',fontsize=20)
            ax.set_ylabel('y', fontsize=20)
            ax.set_zlabel('u', fontsize=20)

            # Scale z-axis according to intensity
            ax.set_zlim(-3*amp,3*amp)

            # Animate 
            initial = ax.plot_surface(X,Y,zarray[:,:,0], cmap=plt.cm.magma, antialiased=False, 
                            rcount=14, ccount=14,
                            vmin=-0.9*amp, vmax=0.9*amp, linewidth=0.005, edgecolor='black')
            plot = [initial];
            
            # Save animated gif to path
            if not os.path.exists(fpath):
                os.makedirs(fpath)

            path = os.path.join(fpath, fname)
            fig.savefig(path+".png")

            def Animate(frn, zarray, plot):
                plot[0].remove()
                surf = ax.plot_surface(X, Y, zarray[:,:,frn],  cmap=plt.cm.magma, antialiased=False, 
                                rcount=int(10+4), ccount=int(10+4),
                                vmin=-0.9*amp, vmax=0.9*amp, linewidth=0.005, edgecolor='black')
                plot[0] = surf

            ani = animation.FuncAnimation(fig, Animate, frn, fargs=(zarray, plot), interval=1000/fps)
            self.frames = ani

       # Save animated gif to path
        if not os.path.exists(fpath):
            os.makedirs(fpath)
    
        path = os.path.join(fpath, fname)
        self.frames.save(path+".mp4", fps=fps)
        
        clip = (VideoFileClip(path+'.mp4'))
        clip.write_gif(path+'.gif',)
        os.remove(path+'.mp4')

    
    def simInitial(self, fname, fpath):
        '''
        Produces graph of drum membrane at t=0 provided initial condition u0
        '''
        N = 50 # Meshsize
        
        r = np.linspace(0, self.radius, N)
        p = np.linspace(0, 2*np.pi, N)
        R, P = np.meshgrid(r, p)
        zarray = np.zeros((N, N, 1))
        X, Y = R*np.cos(P), R*np.sin(P)
        
        zarray[:,:, 0] = self.u0(R,P)
    
        amp = np.amax(zarray[:,:,0])
        
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.title.set_text(fname)
        ax.set_xlabel('x',fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.set_zlabel('u', fontsize=20)
        
        ax.set_zlim(-3*amp,3*amp)
    
        ax.plot_surface(X,Y,zarray[:,:,0], cmap=plt.cm.magma, antialiased=False, 
                            rcount=14, ccount=14,
                            vmin=-1.5*amp, vmax=1.5*amp, linewidth=0.005, edgecolor='black')
        
        # Save animated gif to path
        if not os.path.exists(fpath):
            os.makedirs(fpath)
    
        path = os.path.join(fpath, fname)
        fig.savefig(path+".png")
        
   