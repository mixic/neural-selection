// Exact MiniGrid transitions and partial observations; shared neural implementation.
#include "gpu.cu"
constexpr int MW=16,MH=16,MS=7,MO=512,MC=MS*MS*3*2;
struct MazeObject {int type,color,state,contains;};
struct MazeAgent {int x,y,dir,carrying,target,steps,horizon,through,done;};
struct MazeBatch {
 int count; int *grid=nullptr; MazeObject *objects=nullptr; MazeAgent *agents=nullptr;
 unsigned char *observations=nullptr; float *encoded=nullptr,*lookup=nullptr;
 int *actions=nullptr; double *rewards=nullptr; unsigned char *dones=nullptr;
};
static void destroy_maze(MazeBatch *b){cudaFree(b->grid);cudaFree(b->objects);cudaFree(b->agents);cudaFree(b->observations);cudaFree(b->encoded);cudaFree(b->lookup);cudaFree(b->actions);cudaFree(b->rewards);cudaFree(b->dones);delete b;}
__device__ void maze_observe(MazeBatch b,int i) {
 auto a=b.agents[i];int *world=b.grid+i*MW*MH;auto objects=b.objects+i*MO;
 unsigned char cells[MS*MS*3],clear[MS*MS],visible[MS*MS];
 int topx=a.dir==0?a.x:a.dir==2?a.x-MS+1:a.x-MS/2;
 int topy=a.dir==1?a.y:a.dir==3?a.y-MS+1:a.y-MS/2;
 for(int x=0;x<MS;++x)for(int y=0;y<MS;++y){
  int sx=x,sy=y;for(int k=0;k<a.dir+1;++k){int old=sx;sx=MS-1-sy;sy=old;}
  sx+=topx;sy+=topy;int j=x*MS+y;MazeObject o={2,5,0,-1};
  if(sx>=0&&sx<MW&&sy>=0&&sy<MH){int id=world[sy*MW+sx];o=id<0?MazeObject{1,0,0,-1}:objects[id];}
  cells[j*3]=o.type;cells[j*3+1]=o.color;cells[j*3+2]=o.state;
  clear[j]=o.type!=2 && (o.type!=4 || o.state==0);visible[j]=a.through?1:0;
 }
 if(!a.through){
  visible[(MS/2)*MS+MS-1]=1;
  for(int y=MS-1;y>=0;--y){
   for(int x=0;x<MS-1;++x)if(visible[x*MS+y]&&clear[x*MS+y]){visible[(x+1)*MS+y]=1;if(y>0){visible[(x+1)*MS+y-1]=1;visible[x*MS+y-1]=1;}}
   for(int x=MS-1;x>=1;--x)if(visible[x*MS+y]&&clear[x*MS+y]){visible[(x-1)*MS+y]=1;if(y>0){visible[(x-1)*MS+y-1]=1;visible[x*MS+y-1]=1;}}
  }
 }
 MazeObject carried=a.carrying<0?MazeObject{1,0,0,-1}:objects[a.carrying];int j=(MS/2)*MS+MS-1;
 cells[j*3]=carried.type;cells[j*3+1]=carried.color;cells[j*3+2]=carried.state;
 for(int k=0;k<MS*MS*3;++k){int value=visible[k/3]?cells[k]:0;b.observations[i*MS*MS*3+k]=value;
  b.encoded[i*MC+2*k]=b.lookup[value*MC+2*k];b.encoded[i*MC+2*k+1]=b.lookup[value*MC+2*k+1];}
}
__global__ void maze_observations(MazeBatch b){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<b.count)maze_observe(b,i);}
__global__ void maze_step(MazeBatch b){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=b.count)return;
 auto &a=b.agents[i];b.rewards[i]=0.;if(a.done){b.dones[i]=1;return;}
 int action=b.actions[i];if(action<0){b.dones[i]=1;return;}
 ++a.steps;int dx=a.dir==0?1:a.dir==2?-1:0,dy=a.dir==1?1:a.dir==3?-1:0;
 int x=a.x+dx,y=a.y+dy,p=y*MW+x;int *world=b.grid+i*MW*MH;auto objects=b.objects+i*MO;
 int id=(x>=0&&x<MW&&y>=0&&y<MH)?world[p]:-2;MazeObject o=id>=0?objects[id]:MazeObject{id==-1?1:2,5,0,-1};
 if(action==0)a.dir=(a.dir+3)%4;
 else if(action==1)a.dir=(a.dir+1)%4;
 else if(action==2){
  if(o.type==1||o.type==3||o.type==8||o.type==9||(o.type==4&&o.state==0)){a.x=x;a.y=y;}
  if(o.type==8){a.done=1;b.rewards[i]=1.-0.9*(double(a.steps)/double(a.horizon));}
  if(o.type==9)a.done=1;
 }else if(action==3){
  if(id>=0&&(o.type==5||o.type==6||o.type==7)&&a.carrying<0){a.carrying=id;world[p]=-1;}
  // ObstructedMaze's target is object identity, not its colour/type.
  if(a.carrying>=0&&a.carrying==a.target){a.done=1;b.rewards[i]=1.-0.9*(double(a.steps)/double(a.horizon));}
 }else if(action==4){if(id==-1&&a.carrying>=0){world[p]=a.carrying;a.carrying=-1;}}
 else if(action==5&&id>=0){
  if(o.type==4){if(o.state==2){if(a.carrying>=0&&objects[a.carrying].type==5&&objects[a.carrying].color==o.color)objects[id].state=0;}else objects[id].state=o.state==0?1:0;}
  else if(o.type==7)world[p]=o.contains;
 }
 if(a.steps>=a.horizon)a.done=1;b.dones[i]=a.done;maze_observe(b,i);
}
extern "C" {
void *ns_maze_create(int count,const float *lookup){
 try{if(count<1)throw std::runtime_error("Empty maze batch");auto b=new MazeBatch;b->count=count;
  try{
   check(cudaMalloc((void**)&b->grid,count*MW*MH*sizeof(int)));check(cudaMalloc((void**)&b->objects,count*MO*sizeof(MazeObject)));check(cudaMalloc((void**)&b->agents,count*sizeof(MazeAgent)));
   check(cudaMalloc((void**)&b->observations,count*MS*MS*3));check(cudaMalloc((void**)&b->encoded,count*MC*sizeof(float)));check(cudaMalloc((void**)&b->lookup,11*MC*sizeof(float)));
   check(cudaMalloc((void**)&b->actions,count*sizeof(int)));check(cudaMalloc((void**)&b->rewards,count*sizeof(double)));check(cudaMalloc((void**)&b->dones,count));
   check(cudaMemcpy(b->lookup,lookup,11*MC*sizeof(float),cudaMemcpyHostToDevice));
  }catch(...){destroy_maze(b);throw;}return b;
 }catch(const std::exception &e){error=e.what();return nullptr;}
}
void ns_maze_destroy(void *p){destroy_maze(static_cast<MazeBatch *>(p));}
int ns_maze_reset(void *p,const int *grid,const int *objects,const int *agents){
 try{auto b=*static_cast<MazeBatch *>(p);check(cudaMemcpy(b.grid,grid,b.count*MW*MH*sizeof(int),cudaMemcpyHostToDevice));check(cudaMemcpy(b.objects,objects,b.count*MO*sizeof(MazeObject),cudaMemcpyHostToDevice));check(cudaMemcpy(b.agents,agents,b.count*sizeof(MazeAgent),cudaMemcpyHostToDevice));check(cudaMemset(b.rewards,0,b.count*sizeof(double)));check(cudaMemset(b.dones,0,b.count));maze_observations<<<(b.count+127)/128,128>>>(b);check(cudaGetLastError());check(cudaDeviceSynchronize());return 0;}catch(const std::exception &e){error=e.what();return -1;}
}
int ns_maze_step(void *p,const int *actions){
 try{auto b=*static_cast<MazeBatch *>(p);check(cudaMemcpy(b.actions,actions,b.count*sizeof(int),cudaMemcpyHostToDevice));maze_step<<<(b.count+127)/128,128>>>(b);check(cudaGetLastError());return 0;}catch(const std::exception &e){error=e.what();return -1;}
}
int ns_maze_read(void *p,unsigned char *obs,double *reward,unsigned char *done,int *grid,int *objects,int *agents){
 try{auto b=*static_cast<MazeBatch *>(p);
  if(obs)check(cudaMemcpy(obs,b.observations,b.count*MS*MS*3,cudaMemcpyDeviceToHost));if(reward)check(cudaMemcpy(reward,b.rewards,b.count*sizeof(double),cudaMemcpyDeviceToHost));if(done)check(cudaMemcpy(done,b.dones,b.count,cudaMemcpyDeviceToHost));
  if(grid)check(cudaMemcpy(grid,b.grid,b.count*MW*MH*sizeof(int),cudaMemcpyDeviceToHost));if(objects)check(cudaMemcpy(objects,b.objects,b.count*MO*sizeof(MazeObject),cudaMemcpyDeviceToHost));if(agents)check(cudaMemcpy(agents,b.agents,b.count*sizeof(MazeAgent),cudaMemcpyDeviceToHost));return 0;
 }catch(const std::exception &e){error=e.what();return -1;}
}
int ns_maze_brain_step(void *p,void *brain,const unsigned char *mask,int ticks,float gain,float *output,int *status){
 try{auto m=*static_cast<MazeBatch *>(p);auto &b=*static_cast<GpuBatch *>(brain);if(b.count!=m.count||b.channels!=MC)throw std::runtime_error("Maze/brain dimensions differ");
  check(cudaMemcpy(b.mask,mask,b.count,cudaMemcpyHostToDevice));step_batch<<<b.count,b.threads>>>(b.brains,b.count,b.mask,m.encoded,b.output,ticks,gain,b.status);check(cudaGetLastError());
  check(cudaMemcpy(output,b.output,b.count*b.outputs*sizeof(float),cudaMemcpyDeviceToHost));check(cudaMemcpy(status,b.status,b.count*sizeof(int),cudaMemcpyDeviceToHost));return 0;
 }catch(const std::exception &e){error=e.what();return -1;}
}
}
