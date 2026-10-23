class Geometry:

    def __init__(self, original, states):
        self.original = original
        self.routes = {}
        self.by_route = {}
        self.native = {k: v['native'] for k, v in states.items()}
        options = {}
        for arm, state in states.items():
            grouped = {}
            for row in state['operators']:
                key = (row['rows'], row['kernel'], row['drop'])
                grouped.setdefault(key, []).append(row)
            for (m, kernel, drop), rows in grouped.items():
                hot = statistics.median((r['speedup'] for r in rows))
                cold = statistics.median((r['cold_speedup'] for r in rows))
                if min(hot, cold) < 1.025:
                    continue
                options.setdefault(m, []).append(dict(arm=arm, kernel=kernel, drop=drop, hot=hot, cold=cold))
        for m, rows in options.items():
            self.routes[m] = max(rows, key=lambda r: min(r['hot'], r['cold']))
        specialized = {}
        for row in states['extended']['operators']:
            base = row['base_route']
            key = (row['rows'], base['kernel'], base['drop'], row['kernel'], row['drop'])
            specialized.setdefault(key, []).append(row)
        for (m, base, base_drop, kernel, drop), data in specialized.items():
            hot = statistics.median((r['speedup'] for r in data))
            cold = statistics.median((r['cold_speedup'] for r in data))
            if min(hot, cold) < 1.025:
                continue
            key = (m, base, base_drop)
            route = dict(arm='extended', kernel=kernel, drop=drop, hot=hot, cold=cold)
            if key not in self.by_route or min(hot, cold) > min(self.by_route[key]['hot'], self.by_route[key]['cold']):
                self.by_route[key] = route

    def project(self, x, p, name, drop=0):
        route = self.by_route.get((len(x), name, drop), self.routes.get(len(x)))
        if route is None:
            return self.original.project(x, p, name, drop)
        return self.native[route['arm']].project(x, p, route['kernel'], route['drop'])
