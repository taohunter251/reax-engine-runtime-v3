class Combined:

    def __init__(self, natives):
        self.natives = natives

    def project(self, x, p, name, drop=0):
        arm, kernel = name.split('::', 1)
        return self.natives[arm].project(x, p, kernel, drop)
