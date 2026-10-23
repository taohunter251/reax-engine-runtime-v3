"""Canonical suffix repairs with UID240's scheduler and MPF delivery batching."""
import collections
import threading
import time

import numpy as np
import torch

from s1eng.model import BatchLoop


class Leases:
    def __init__(self, rows, count):
        self.buffers = [torch.empty((2, rows, 2560), device='cuda', dtype=torch.bfloat16)
                        for _ in range(count)]
        self.free = collections.deque(range(count))
        self.users = {}
        self.lock = threading.Lock()

    def take(self, users):
        with self.lock:
            index = self.free.popleft()
            self.users[index] = users
            return index, self.buffers[index]

    def release(self, index):
        if index is None:
            return
        with self.lock:
            self.users[index] -= 1
            if not self.users[index]:
                del self.users[index]
                self.free.append(index)


class GuardLoop(BatchLoop):
    def __init__(self, model, fast_lanes, repair_lane, checkpoints, repair_inputs, front, margin, depth):
        self.fast_lanes = list(fast_lanes)
        self.repair_lane = repair_lane
        self.checkpoints, self.repair_inputs = checkpoints, repair_inputs
        self.margin = margin
        self.mode = 'combined'
        self.native_control = None
        self.repairs = self.completed_exact = 0
        self.lease_pool = Leases(max(len(pair[0]) for pair in checkpoints.values()), 32)
        self.ready_events = {index: torch.cuda.Event() for index in range(32)}
        self.delivery_tag = front.tag
        # A repair is an internal continuation. Group result writes only when the
        # final answers are known, so a retry never blocks unrelated MPF replies.
        front.tag = lambda jobs: None
        self.delivery_front = front
        # Gate the unchanged native fast lanes first. Repair graphs consume saved
        # hidden states, so comparing them to a full-token forward is meaningless.
        super().__init__(model, list(fast_lanes), depth=depth)
        self.lanes.append(repair_lane)
        self.done_q.append(collections.deque())
        self.done_cv.append(threading.Condition())
        threading.Thread(target=self._completer, args=(repair_lane.idx,),
                         name='canonical-repair-done', daemon=True).start()

    def submit(self, job):
        if self.mode == 'uid240':
            return self.native_control.submit(job)
        return super().submit(job)

    def _choose(self):
        queue, lanes = self.q, self.lanes
        deadlines = []
        try:
            for exact, allowed in ((True, [self.repair_lane]), (False, self.fast_lanes)):
                jobs = [j for j in queue if j.exact == exact]
                if not jobs:
                    continue
                self.q, self.lanes = type(queue)(jobs), allowed
                choice = super()._choose()
                if choice is not None:
                    return choice
                if self.wake_at is not None:
                    deadlines.append(self.wake_at)
            self.wake_at = min(deadlines) if deadlines else None
        finally:
            self.q, self.lanes = queue, lanes

    def _launch(self, lane, length, jobs, slot, packed):
        if jobs[0].exact:
            assert lane is self.repair_lane and packed is not None
            states = self.repair_inputs[id(packed[1][1])]
            with torch.cuda.stream(lane.stream):
                for tensor in states:
                    tensor.zero_()
                offset = 0
                for job in jobs:
                    lane.stream.wait_event(self.ready_events[job.lease])
                    count = len(job.checkpoint[0])
                    for destination, source in zip(states, job.checkpoint):
                        destination[offset:offset+count].copy_(source)
                    offset += count
            return BatchLoop._launch(self, lane, length, jobs, slot, packed)

        BatchLoop._launch(self, lane, length, jobs, slot, packed)
        if packed is None:
            return
        entry = packed[-1] if packed[0] == 'mx' else packed[1]
        state = self.checkpoints.get(id(entry[1]))
        if state is None:
            return
        lease, destination = self.lease_pool.take(len(jobs))
        with torch.cuda.stream(lane.stream):
            rows = len(state[0])
            destination[0, :rows].copy_(state[0])
            destination[1, :rows].copy_(state[1])
            offset = 0
            for job in jobs:
                count = job.n - (job.pl if entry[2].pfx else 0)
                job.checkpoint = tuple(destination[t, offset:offset+count] for t in range(2))
                job.lease = lease
                offset += count
            self.ready_events[lease].record(lane.stream)
            slot.event.record(lane.stream)

    def _completer(self, index):
        queue, condition = self.done_q[index], self.done_cv[index]
        while True:
            with condition:
                while not queue:
                    condition.wait()
                jobs, slot, lane = queue.popleft()
            slot.event.synchronize()
            values = slot.out[:len(jobs)].float().numpy().copy()
            final, retry = [], []
            for row, job in enumerate(jobs):
                uncertain = False
                if not job.exact and job.checkpoint is not None and job.option_count > 1:
                    logits = np.sort(values[row, :job.option_count])
                    uncertain = float(logits[-1] - logits[-2]) <= self.margin
                if uncertain:
                    job.exact = True
                    job.t_enq = time.perf_counter()
                    retry.append(job)
                    self.repairs += 1
                else:
                    self.completed_exact += int(job.exact)
                    self.lease_pool.release(job.lease)
                    job.lease = job.checkpoint = None
                    final.append((job, values[row]))
            if self.trace_every and slot.tag is not None:
                self._trace_end(slot, jobs)
            with self.cv:
                lane.busy -= 1
                lane.t_free = time.perf_counter()
                self.free.append(slot)
                self.q.extend(retry)
                self.back += len(final)
                self.t_back = lane.t_free
                if self.bb_ledger:
                    self.bb_ret.append([lane.t_free, len(final)])
                if self.arms:
                    self.tr_led.append([lane.t_free, len(final)])
                self.cv.notify()
            self.delivery_tag([job for job, _ in final])
            for job, value in final:
                job.done(value)
