
class Approx:

    def get_props(self):
        raise NotImplementedError("subclass should implement this")

    def get_equations(self, dest, sources, derv=0):
        raise NotImplementedError("subclass should implement this")

