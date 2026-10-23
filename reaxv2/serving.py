import collections
import contextvars
import json
import threading
import time
import numpy as np
import torch
from s1eng.model import BatchLoop, Job, Lane
from s1eng.server import Engine, _resolve
OPTION_COUNT = contextvars.ContextVar('option_count', default=16)
_PENDING = contextvars.ContextVar('arrival_registration', default=False)

class TrackedJob(Job):
    __slots__=('option_count','exact','checkpoint')
    def __init__(self,ids,done,option_count=16,exact=False,checkpoint=None):
        super().__init__(ids,done)
        self.option_count,self.exact,self.checkpoint=option_count,exact,checkpoint


class RepairEngine(Engine):
    async def decide(self,headers,body,tm=None):
        from s1eng.prompt import options_of
        count=16
        try:count=len(options_of(next(iter(json.loads(body)['questions'].values()))))
        except (ValueError,KeyError,TypeError,StopIteration):pass
        token=OPTION_COUNT.set(count)
        try:return await super().decide(headers,body,tm)
        finally:OPTION_COUNT.reset(token)
    def _run(self,ids):
        loop=self.loop;future=loop.create_future()
        def done(value):loop.call_soon_threadsafe(_resolve,future,value)
        self.gpu.submit(TrackedJob(np.asarray(ids,dtype=np.int64),done,OPTION_COUNT.get()))
        return future


class RepairLoop(BatchLoop):
    def __init__(self,model,fast_lanes,repair_lane,checkpoints,inputs,depth):
        self.fast_lanes=list(fast_lanes);self.repair_lane=repair_lane
        self.checkpoints,self.repair_inputs=checkpoints,inputs
        self.enabled=False;self.margin=.25;self.repairs=0;self.completed_exact=0
        super().__init__(model,[*fast_lanes,repair_lane],depth=depth)
    def submit(self,job):
        if not job.exact:
            original=job.done
            def done(value):
                if self.enabled and job.checkpoint is not None and not isinstance(value,BaseException):
                    values=np.sort(value[:job.option_count])
                    gap=float(values[-1]-values[-2]) if job.option_count>1 else float('inf')
                    if gap<=self.margin:
                        retry=TrackedJob(job.ids,original,job.option_count,True,job.checkpoint)
                        self.repairs+=1
                        BatchLoop.submit(self,retry)
                        return
                original(value)
            job.done=done
        BatchLoop.submit(self,job)
    def _choose(self):
        queue,lanes=self.q,self.lanes
        deadlines=[]
        try:
            for exact,allowed in ((True,[self.repair_lane]),(False,self.fast_lanes)):
                jobs=[j for j in queue if j.exact==exact]
                if not jobs:continue
                self.q=type(queue)(jobs);self.lanes=allowed
                pick=super()._choose()
                if pick is not None:return pick
                if self.wake_at is not None:deadlines.append(self.wake_at)
            self.wake_at=min(deadlines) if deadlines else None
        finally:self.q,self.lanes=queue,lanes
    def _launch(self,lane,L,jobs,slot,pk):
        if jobs[0].exact:
            assert pk is not None and lane is self.repair_lane
            _,entry=pk;states=self.repair_inputs[id(entry[1])]
            with torch.cuda.stream(lane.stream):
                for tensor in states:tensor.zero_()
                offset=0
                for job in jobs:
                    rows=len(job.checkpoint[0])
                    for destination,source in zip(states,job.checkpoint):
                        destination[offset:offset+rows].copy_(source)
                    offset+=rows
            super()._launch(lane,L,jobs,slot,pk)
            self.completed_exact+=len(jobs)
            return
        super()._launch(lane,L,jobs,slot,pk)
        if not self.enabled or pk is None:return
        _,entry=pk
        source=self.checkpoints.get(id(entry[1]))
        if source is None:return
        with torch.cuda.stream(lane.stream):
            retained=tuple(t.clone() for t in source)
            offset=0
            for job in jobs:
                rows=job.n-(job.pl if entry[2].pfx else 0)
                job.checkpoint=tuple(t[offset:offset+rows] for t in retained)
                offset+=rows
            slot.event.record()

class LeasedJob(TrackedJob):
    __slots__=('lease',)
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.lease=None


class BufferPool:
    def __init__(self,rows,count=32):
        self.buffers=[torch.empty((2,rows,2560),device='cuda',dtype=torch.bfloat16) for _ in range(count)]
        self.free=collections.deque(range(count));self.users={};self.lock=threading.Lock()
    def take(self,users):
        with self.lock:
            index=self.free.popleft();self.users[index]=users
            return index,self.buffers[index]
    def release(self,index):
        if index is None:return
        with self.lock:
            self.users[index]-=1
            if not self.users[index]:del self.users[index];self.free.append(index)


class Loop(RepairLoop):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.buffers=BufferPool(max(len(pair[0]) for pair in self.checkpoints.values()))
        self.guard_kind='fixed'
    def uncertain(self,value,count):
        if count <= 1:
            return False
        values = np.sort(value[:count])
        return float(values[-1]-values[-2]) <= self.margin
    def submit(self,job):
        if not job.exact:
            original=job.done
            def done(value):
                checkpoint,lease=job.checkpoint,job.lease
                job.done=original;job.checkpoint=job.lease=None
                if self.enabled and checkpoint is not None and not isinstance(value,BaseException) and self.uncertain(value,job.option_count):
                    def corrected(result):
                        self.buffers.release(lease);original(result)
                    retry=LeasedJob(job.ids,corrected,job.option_count,True,checkpoint)
                    self.repairs+=1
                    BatchLoop.submit(self,retry)
                else:
                    self.buffers.release(lease);original(value)
            job.done=done
        BatchLoop.submit(self,job)
    def _launch(self,lane,L,jobs,slot,pk):
        if jobs[0].exact:return super()._launch(lane,L,jobs,slot,pk)
        BatchLoop._launch(self,lane,L,jobs,slot,pk)
        if not self.enabled or pk is None:return
        _,entry=pk;source=self.checkpoints.get(id(entry[1]))
        if source is None:return
        lease,buffer=self.buffers.take(len(jobs));rows=len(source[0])
        with torch.cuda.stream(lane.stream):
            buffer[0,:rows].copy_(source[0]);buffer[1,:rows].copy_(source[1])
            offset=0
            for job in jobs:
                count=job.n-(job.pl if entry[2].pfx else 0)
                job.checkpoint=tuple(buffer[t,offset:offset+count] for t in range(2));job.lease=lease
                offset+=count
            slot.event.record()


def exact_lane(gpu,model,output,stream_priority=0):
    from s1eng import kprio
    from s1eng.batchfwd import PackedMeta,packed_layout
    f=gpu.m.pf;previous=f.bf.ws
    workspace=torch.empty(max(1<<20,f.pinned.ws_bytes),device='cuda',dtype=torch.uint8)
    stream,cap=torch.cuda.Stream(priority=stream_priority),torch.cuda.Stream();pool=torch.cuda.graph_pool_handle()
    packed={};inputs={};count=0
    try:
        f.bf.ws=workspace
        for L in (128,192,256,320):
            prefix=gpu.lanes[0].pfx.get(L,0)
            for B in (1,2,4):
                limit=B*(L-prefix)
                for rows in [*range(32,limit,32),limit]:
                    meta=PackedMeta(B,L,rows,pfx=prefix)
                    length=min(L,prefix+rows//B)
                    st,sl,rp,last=packed_layout([length]*B,B,L,rows,prefix)
                    for name,array in (('start',st),('slen',sl),('row_pos',rp),('last',last)):
                        getattr(meta,name).copy_(torch.from_numpy(array))
                    states=tuple(torch.zeros((rows,2560),device='cuda',dtype=torch.bfloat16) for _ in range(2))
                    with torch.cuda.stream(stream):model.suffix(meta,*states,False)
                    stream.synchronize()
                    graph=torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph,pool=pool,stream=cap):result=model.suffix(meta,*states,False)
                    kprio.apply(graph,kprio.mode());graph.instantiate()
                    inputs[id(graph)]=states;packed.setdefault((L,B),[]).append((rows,graph,meta,result));count+=1
                print('TIGHT_REPAIR_CAPTURE '+str((L,B,count)),flush=True)
        lane=Lane(2,stream,{}, {},packed,ws=[workspace],cap_stream=cap,pfx=gpu.lanes[0].pfx)
        torch.cuda.synchronize()
        print('EXACT_GRAPHS '+str(count),flush=True)
        return lane,inputs
    finally:f.bf.ws=previous

class ReadyLoop(Loop):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ready={buffer.untyped_storage().data_ptr():torch.cuda.Event() for buffer in self.buffers.buffers}
    def _launch(self,lane,L,jobs,slot,pk):
        if jobs[0].exact:
            with torch.cuda.stream(lane.stream):
                for job in jobs:lane.stream.wait_event(self.ready[job.checkpoint[0].untyped_storage().data_ptr()])
            return super()._launch(lane,L,jobs,slot,pk)
        super()._launch(lane,L,jobs,slot,pk)
        if self.enabled and pk is not None:
            event=self.ready[jobs[0].checkpoint[0].untyped_storage().data_ptr()]
            with torch.cuda.stream(lane.stream):event.record()

class ArrivalLoop(ReadyLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arrivals = 0
        self.returning_clients = 0
        self.last_delivery = 0.
        self.return_window = .010

    def _choose(self):
        old = self.hold_n
        finished = self.returning_clients == 0 or time.perf_counter()-self.last_delivery > self.return_window
        if self.arrivals == 0 and finished:
            self.hold_n = 0
        try:
            return super()._choose()
        finally:
            self.hold_n = old


class ArrivalEngine(RepairEngine):
    async def decide(self, headers, body, tm=None):
        with self.gpu.cv:
            self.gpu.arrivals += 1
        token = _PENDING.set(True)
        try:
            return await super().decide(headers, body, tm)
        finally:
            if _PENDING.get():
                with self.gpu.cv:
                    self.gpu.arrivals -= 1
            _PENDING.reset(token)

    def _run(self, ids):
        event_loop = self.loop
        future = event_loop.create_future()
        if _PENDING.get():
            with self.gpu.cv:
                self.gpu.arrivals -= 1
                self.gpu.returning_clients = max(0, self.gpu.returning_clients-1)
            _PENDING.set(False)
        def done(value):
            # RepairLoop calls this only after the final fast or corrected answer.
            with self.gpu.cv:
                self.gpu.returning_clients += 1
                self.gpu.last_delivery = time.perf_counter()
            event_loop.call_soon_threadsafe(_resolve, future, value)
        self.gpu.submit(LeasedJob(np.asarray(ids,dtype=np.int64),done,OPTION_COUNT.get()))
        return future

