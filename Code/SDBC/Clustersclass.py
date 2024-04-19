# cluster 类
class Cluster(object):
    def __init__(self, C_id):
        self.C_id = C_id
        self.members = list()
        self.OX = 0
        self.DX = 0
        self.OY = 0
        self.DY = 0
        self.neighs = list()
        self.GR = 0
        # 标准化的density
        self.nor_density = 0
        self.label = 0


