# 流类
class OD_Flow(object):
    # def __init__(self, f_id, ox, oy, otime, dx, dy, dtime):
    def __init__(self, f_id, ox, oy, dx, dy):
        self.f_id = f_id
        self.OX = ox
        self.OY = oy
        self.DX = dx
        self.DY = dy
        # self.Otime = otime
        # self.Dtime = dtime
        self.neighs = list()
        self.neighs_id = list()
        self.density = 0
        #self.wij = 0  # 指示函数
        self.Gi = 0
        self.nor_density = 0
        self.index = 0
