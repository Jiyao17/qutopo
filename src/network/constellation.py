
import os
import enum

class ConstellationOpt(enum.Enum):
    GPS = 1
    GLONASS = 2
    GALILEO = 3
    BEIDOU = 4

CONSTELLATION_FILENAME = {
    ConstellationOpt.GPS: 'constellation/gps.tle',
    ConstellationOpt.GLONASS: 'constellation/glonass.tle',
    ConstellationOpt.GALILEO: 'constellation/galileo.tle',
    ConstellationOpt.BEIDOU: 'constellation/beidou.tle'
}

class ConstellationPosition():
    @staticmethod
    def get_constellation(path: str):
        pass

    def __init__(self,
            constellation: ConstellationOpt = ConstellationOpt.GPS,
        ) -> None:
        super().__init__()
        
        filename = CONSTELLATION_FILENAME[constellation]
        path = os.path.join(os.path.dirname(__file__), filename)
        
        self.constellation = ConstellationPosition.get_constellation(path)

    def get_positions(self):
        pass