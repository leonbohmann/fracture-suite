class CrackSystem:
    distance: list[float]
    "Distance from the center of the specimen."
    time: list[float]
    "Time in ns."
    status: list[float]
    "Status of the crack system."
    positions: list[tuple[float, float]]
    "Positions of the crack tips."

    def __init__(self, clr):
        self.clr = clr
        self.time = []
        self.distance = []
        self.status = []
        self.positions = []


    def advance(self, ):
        pass