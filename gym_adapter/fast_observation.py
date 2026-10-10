"""Exact compiled partial-view construction for two immutable-object MiniGrid tasks."""
import ctypes as ct
from types import MethodType
import numpy as np
from .adapter import native

SUPPORTED={'MiniGrid-Empty-Random-6x6-v0','MiniGrid-MemoryS7-v0'}

def install(env):
    base=env.unwrapped;lib=native();ptr=ct.POINTER(ct.c_ubyte)
    lib.ns_grid_observation.argtypes=[ptr,ptr]+[ct.c_int]*8+[ptr,ptr]
    # These two tasks have immutable object encodings/opacity. Pick-up and drop
    # move object references; reset constructs a new grid and clears the cache.
    grid_ref=None;cache={None:((1,0,0),1)}
    def gen_obs(self):
        nonlocal grid_ref,cache
        if grid_ref is not self.grid:
            grid_ref=self.grid;cache={None:((1,0,0),1)}
        world=np.empty((len(self.grid.grid),3),np.uint8);clear=np.empty(len(world),np.uint8)
        for i,cell in enumerate(self.grid.grid):
            if cell not in cache:cache[cell]=(cell.encode(),int(cell.see_behind()))
            encoded,visible=cache[cell];world[i]=encoded;clear[i]=visible
        carrying=np.asarray(self.carrying.encode() if self.carrying else (1,0,0),np.uint8)
        size=self.agent_view_size;out=np.empty((size,size,3),np.uint8);topx,topy,_,_=self.get_view_exts()
        lib.ns_grid_observation(world.ctypes.data_as(ptr),clear.ctypes.data_as(ptr),self.grid.width,self.grid.height,topx,topy,size,int(self.agent_dir),int(self.see_through_walls),0,carrying.ctypes.data_as(ptr),out.ctypes.data_as(ptr))
        return {'image':out,'direction':self.agent_dir,'mission':self.mission}
    base.gen_obs=MethodType(gen_obs,base)
