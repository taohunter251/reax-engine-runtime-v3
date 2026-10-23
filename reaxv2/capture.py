import torch

def capture(gpu,model,output):
    from s1eng import kprio
    from s1eng.batchfwd import PackedMeta,packed_layout
    f=gpu.m.pf;ws=f.bf.ws
    
    candidates=[];checkpoints={};counts=[]
    try:
        for lane in gpu.lanes:
            f.bf.ws=lane.ws[0];pool=torch.cuda.graph_pool_handle();packed={}
            entries=[(L,B,entry) for (L,B),items in lane.packed.items() for entry in items]
            for index,(L,B,entry) in enumerate(sorted(entries,key=lambda x:-x[2][0])):
                rows,_,meta,_=entry
                with torch.cuda.stream(lane.stream):model(meta,True)
                lane.stream.synchronize()
                graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph,pool=pool,stream=lane.cap_stream):
                    result,state=model(meta,True)
                priorities=kprio.apply(graph,kprio.mode());graph.instantiate()
                checkpoints[id(graph)]=state
                packed.setdefault((L,B),[]).append((rows,graph,meta,result))
                counts.append({'lane':lane.idx,'length':L,'batch':B,'rows':rows,'priorities':priorities})
                if index%64==0:print('ASYNC_CAPTURE '+str((lane.idx,index,len(entries))),flush=True)
            for entries in packed.values():entries.sort(key=lambda x:x[0])
            candidates.append(packed)
        torch.cuda.synchronize()
        return candidates, checkpoints
    finally:f.bf.ws=ws

