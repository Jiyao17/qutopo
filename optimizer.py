
import copy
from abc import ABC, abstractmethod

import gurobipy as gp
from gurobipy import GRB

import numpy as np

from net import NodeID, NodePair, EdgeTuple, Path
from net import BufferedNode, VEdge, QON, QONTask, Task
import quantum


class ObjType:
    FEASIBILITY = 0
    MAX_THROUGHPUT = 1
    MIN_LATENCY = 2


class Optimizer(ABC):

    def __init__(self, qon_task: QONTask) -> None:
        self.qon_task = copy.deepcopy(qon_task)
        # dump all parameters needed by optimizer to a dict
        self.params = {}
        self.qon = copy.deepcopy(qon_task.qon)

        self.model = gp.Model("LP")

    def import_params(self, ) -> dict:
        """
        import all parameters needed to optimizer
        same notation as Table 1 in the paper
        """
        self.params = {}

        self.params['T']: int = self.qon_task.time_slots
        # time slots number
        self.T = self.params['T']
        # duration of each time slot, in seconds
        self.params['delta']: float = self.qon_task.delta
        self.Delta = self.params['delta']

        # all edges, [EdgeTuple]
        self.params['E']: 'list[EdgeTuple]' = []
        # all real edges, [EdgeTuple]
        self.params['rE']: 'list[EdgeTuple]' = []
        # all virtual edges, [EdgeTuple]
        self.params['vE']: 'list[EdgeTuple]' = []
        # capacity of each edge, {e: capacity}
        self.params['c']: 'dict[EdgeTuple, int]' = {}
        for (u, v, k, ddict) in self.qon_task.qon.vnet.edges.data(keys=True):
            edge_obj: VEdge = ddict['obj']
            edge_tuple: EdgeTuple = (u, v, k)
            self.params['E'].append(edge_tuple)
            if edge_obj.vlink:
                self.params['vE'].append(edge_tuple)
            else:
                self.params['rE'].append(edge_tuple)
            self.params['c'][edge_tuple] = edge_obj.capacity
        self.E = self.params['E']
        self.rE = self.params['rE']
        self.vE = self.params['vE']
        self.c = self.params['c']
        # user pairs, [(src, dst)]
        self.params['K']: 'list[NodePair]' = copy.deepcopy(self.qon_task.user_pairs)
        self.K = self.params['K']
        # real paths between user pairs, {k: [path]}
        self.params['PKN']: 'dict[NodePair, list[Path]]' = copy.deepcopy(self.qon_task.up_rpaths)
        self.PKN = self.params['PKN']
        # virtual paths between user pairs, {k: [path]}
        self.params['PKS']: 'dict[NodePair, list[Path]]' = copy.deepcopy(self.qon_task.up_vpaths)
        self.PKS = self.params['PKS']
        # merge PkN and PkS
        self.params['PKNS']: 'dict[NodePair, list[Path]]' = {}
        for k in self.K:
            self.params['PKNS'][k] = self.params['PKN'][k] + self.params['PKS'][k]
        self.PKNS = self.params['PKNS']

        # QMs
        self.params['S']: list[NodeID] = copy.deepcopy(self.qon_task.qon.QMs)
        self.S = self.params['S']
        # storage pairs, [(src, dst)]
        self.params['J']: 'list[NodePair]' = copy.deepcopy(self.qon_task.qon.qm_pairs)
        self.J = self.params['J']


        # real paths between storage pairs, {j: [path]}
        self.params['PJN']: 'dict[NodePair, list[Path]]' = copy.deepcopy(self.qon_task.qon.qm_rpaths)
        self.PJN = self.params['PJN']
        # virtual paths between storage pairs, {j: [path]}
        self.params['PJS']: 'dict[NodePair, list[Path]]' = copy.deepcopy(self.qon_task.qon.qm_vpaths)
        self.PJS = self.params['PJS']
        # merge PJN and PJS
        self.params['PJNS']: 'dict[NodePair, list[Path]]' = {}
        for j in self.J:
            self.params['PJNS'][j] = self.params['PJN'][j] + self.params['PJS'][j]
        self.PJNS = self.params['PJNS']
        # merge PKNS and PJNS
        self.params['PKJNS']: 'dict[NodePair, list[Path]]' = self.PKNS | self.PJNS
        self.PKJNS = self.params['PKJNS']
        # merge PKS and PJS
        self.params['PKJS']: 'dict[NodePair, list[Path]]' = self.PKS | self.PJS
        self.PKJS = self.params['PKJS']
        # merge K and J
        self.params['KJ']: 'list[NodePair]' = self.params['K'] + self.params['J']
        self.KJ = self.params['KJ']
        # workload of user pair, {(k, t): load}
        self.params['D']: 'dict[tuple[NodePair, int], int]' = copy.deepcopy(self.qon_task.workload)
        self.D = self.params['D']
        # storage capacity of QM, {s: capacity}
        self.params['B']: 'dict[NodeID, int]' = {}
        for qm in self.params['S']:
            node_obj: Buffered]Node = self.qon_task.qon.vnet.nodes[qm]['obj']
            self.params['B'][qm] = node_obj.storage
        self.B = self.params['B']
        # basic fidelity of path, {p: fidelity}
        self.params['Fp']: 'dict[Path, float]' = {}
        for k, PkNS in self.params['PKJNS'].items():
            for p in PkNS:
                self.params['Fp'][p], cap = QON.swap_along_path(self.qon_task.qon.vnet, p)
        self.Fp = self.params['Fp']
                # fidelity threshold of user pair, {(k, t): fidelity}
        self.params['F']: 'dict[tuple[NodePair, int], float]' = copy.deepcopy(self.qon_task.fid_req)
        for j in self.J:
            for t in range(self.T):
                if (j, t) not in self.params['F']:
                    self.params['F'][j, t] = 0
        self.F = self.params['F']
        # Entanglement generation rate, object value
        # (NodePair, Path, time) -> EPR generation rate
        self.params['w']: 'dict[tuple[NodePair, Path, int], int]' = {}
        for t in range(self.qon_task.time_slots):
            for k, PkNS in self.PKJNS.items():
                for p in PkNS:
                    self.params['w'][k, p, t] = 0
        self.w = self.params['w']
        
        # storage evolution
        # (NodePair, Path, time) -> EPR # in storage
        self.params['u']: 'dict[tuple[NodePair, Path, int], int]' = {}
        for t in range(self.qon_task.time_slots):
            for u, PjNS in self.params['PJNS'].items():
                for p in PjNS:
                    self.params['u'][u, p, t] = 0
        self.u = self.params['u']

        # path length
        self.params['Pl']: 'dict[Path, int]' = {}
        for np, paths in self.PKJNS.items():
            for path in paths:
                self.params['Pl'][path] = len(path)
        self.Pl = self.params['Pl']

        return self.params

    @abstractmethod
    def add_constrs(self, cs: str='') -> None:
        pass
    
    @abstractmethod
    def optimize(self, objective: ObjType=ObjType.FEASIBILITY) -> float:
        pass


class QONOptm(Optimizer):

    def add_constrs(self, cs: str='') -> None:
        def add_init_constrs():
            """
            This function basically adds constraints 2, 7, 8 in the paper:
            init w and u for t = 0, build u from w
            add constraints on w and u (actually also w)
            """
            # init all w[k, p, 0] to 0
            for k in self.KJ:
                for p in self.PKJNS[k]:
                    self.model.addConstr(self.w_[k, p, 0] == 0, name=f"w_init_{k}_{p}")
            
            # constrain 7: w[k, p, t] >= 0
            for t in range(0, self.T):
                for k in self.KJ:
                    for p in self.PKJNS[k]:
                        name = f"c7_{k}_{p}_{t}"
                        self.model.addConstr(
                            self.w_[k, p, t] >= 0,
                            name=name
                        )

            # constrain 8 (& 2): u[j, p, t] >= 0
            # init all u[j, ps, 0] to 0
            # set all u[j, ps, t] for t > 0
            for j in self.J:
                    for ps in self.PJNS[j]:
                        self.u_[j, ps, 0] = 0
            for t in range(1, self.T):
                for j in self.J:
                    for ps in self.PJNS[j]:
                        KJ_ = copy.deepcopy(self.KJ)
                        KJ_.remove(j)
                        # prepare c2_right_sum for c2_right
                        c2_right_sum = gp.LinExpr(0)
                        for k in KJ_:
                            for p in self.PKJS[k]:
                                if QON.is_subpath(self.qon_task.qon.vnet, ps, p, 2):
                                    g = quantum.calc_epr_num(self.Fp[p], self.F[k, t-1])
                                    term = g * self.Delta * self.w_[k, p, t-1]
                                    c2_right_sum += term

                        c2_right = self.u_[j, ps, t-1] - c2_right_sum \
                            + self.w_[j, ps, t-1] * self.Delta
                        self.u_[j, ps, t] = c2_right
                        self.model.addConstr(self.u_[j, ps, t] >= 0,
                            name=f"u_{j}_{ps}_{t}")

        def add_c3():
            for t in range(1, self.T):
                for j in self.J:
                    for ps in self.PJNS[j]:
                        name = f"c3_{j}_{ps}_{t}"
                        c3_left = gp.LinExpr(0)
                        KJ_ = copy.deepcopy(self.KJ)
                        KJ_.remove(j)
                        for k in KJ_:
                            for p in self.PKJS[k]:
                                if QON.is_subpath(self.qon_task.qon.vnet, ps, p, 2):
                                    g = quantum.calc_epr_num(self.Fp[p], self.F[k, t])
                                    c3_left += self.w_[k, p, t] * self.Delta * g
                        
                        self.model.addConstr(c3_left <= self.u[j, ps, t], name=name)

        def add_c4():
            for t in range(self.T):
                for k in self.K:
                    c4_left = gp.LinExpr(0)
                    for p in self.PKNS[k]:
                        c4_left += self.w_[k, p, t] 
                    name = f"c4_{k}_{t}"
                    self.model.addConstr(
                        c4_left == self.D[k, t],
                        name=name
                    )
        
        def add_c5():
            for t in range(self.T):
                for edge_tuple in self.rE:
                    name = f"c5_{edge_tuple}_{t}"
                    c5_left = gp.LinExpr(0)
                    for k in self.KJ:
                        for p in self.PKJNS[k]:
                            if edge_tuple in p:
                                g = quantum.calc_epr_num(self.Fp[p], self.F[k, t])
                                c5_left += self.w_[k, p, t] * g

                    self.model.addConstr(c5_left <= self.c[edge_tuple],
                        name=name)

        def add_c6():
            for t in range(1, self.T):
                for s in self.S:
                    name = f"c6_{s}_{t}"
                    c6_left = gp.LinExpr(0)
                    for s2 in self.S:
                        if (s, s2) in self.J:
                            j = (s, s2)
                            for ps in self.PJNS[j]:
                                c6_left += self.u_[j, ps, t]

                    self.model.addConstr(c6_left <= self.B[s], name=name)

        # add decision variables
        w_keys = gp.tuplelist(self.w.keys())
        w_names = [f"w_{k}_{p}_{t}" for k, p, t in w_keys]
        # u_keys = gp.tuplelist(self.u.keys())
        # u_names = [f"u_{j}_{ps}_{t}" for j, ps, t in u_keys]
        # w_kpt, constraint 7
        self.w_ = self.model.addVars(w_keys, vtype=GRB.INTEGER, name=w_names)
        # u_kpt, temp variable used for constraints, not decision variables
        self.u_: 'dict[tuple[NodePair, Path, int], gp.LinExpr]' = {}
        # self.u_ = self.model.addVars(u_keys, vtype=GRB.CONTINUOUS, name=u_names)

        add_init_constrs()

        for c in cs:
            if c == '3':
                add_c3()
            elif c == '4':
                add_c4()
            elif c == '5':
                add_c5()
            elif c == '6':
                add_c6()
            else:
                raise ValueError(f"unknown constraint: {c}")
        
        self.model.update()

    def del_constrs(self, cs: str='') -> None:
        for c in cs:
            if c == '2':
                for t in range(1, self.T):
                    for j in self.J:
                        for ps in self.PJNS[j]:
                            name = f"c2_{j}_{ps}_{t}"
                            self.model.remove(self.model.getConstrByName(name))
            elif c == '3':
                for t in range(self.T):
                    for j in self.J:
                        for ps in self.PJNS[j]:
                            name = f"c3_{j}_{ps}_{t}"
                            self.model.remove(self.model.getConstrByName(name))
            elif c == '4':
                for t in range(1, self.T):
                    for k in self.K:
                        name = f"c4_{k}_{t}"
                        self.model.remove(self.model.getConstrByName(name))
            elif c == '5':
                for t in range(self.T):
                    for edge_tuple in self.rE:
                        name = f"c5_{edge_tuple}_{t}"
                        self.model.remove(self.model.getConstrByName(name))
            elif c == '6':
                for t in range(self.T):
                    for s in self.S:
                        name = f"c6_{s}_{t}"
                        self.model.remove(self.model.getConstrByName(name))
            else:
                raise ValueError(f"unknown constraint: {c}")

        self.model.update()

    def optimize(self, objective: ObjType=ObjType.FEASIBILITY) -> float:
        if objective == ObjType.FEASIBILITY:
            self.model.setObjective(1, GRB.MINIMIZE)
        elif objective == ObjType.MAX_THROUGHPUT:
            # remove all constraints in c4
            self.del_constrs('4')
            # set objective
            obj_expr = gp.LinExpr(0)
            for t in range(self.T):
                for k in self.K:
                    for p in self.PKNS[k]:
                        obj_expr += self.w_[k, p, t]
            self.model.setObjective(obj_expr, GRB.MAXIMIZE)

        elif objective == ObjType.MIN_LATENCY:
            obj_expr = gp.LinExpr(0)
            path_num = 0
            for t in range(self.T):
                for k in self.K:
                    for p in self.PKNS[k]:
                        # obj_expr += self.w_[k, p, t]*(len(p) - 1)
                        # obj_expr += self.w_[k, p, t] * (self.Pl[p] - 1)
                        obj_expr += self.Pl[p] - 1
                        path_num += 1
            obj_expr /= path_num
            self.model.setObjective(obj_expr, GRB.MINIMIZE)

        else:
            raise ValueError
        
        self.model.update()
        # model size
        print(f"model size: {self.model.NumVars}, {self.model.NumConstrs}")
        self.model.optimize()
        return self.model.ObjVal


class RouteAlgo(Optimizer):
    """
    Central Controller of a Quantum Overlay Network
    (Routing Algorithm)
    1. path = decisions in (E, T)
    2. prfy = purification on path
    3. plcm = QM placement
    """

    def __init__(self, qon_task: QONTask) -> None:
        super().__init__(qon_task)
    

class CoPathAlgo():

    def __init__(self, task: Task) -> None:
        pass

    



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    np.random.seed(0)

    qon = QON(QM_num=3)
    qon.set_QMs_EUs(QON.QMSelectMethod.MAX_DEGREE)
    qon.rnet_gen()
    # qon.rnet_gen(100, (200, 1401),)
    qon.vnet_gen(5)

    task = QONTask(qon, 5)
    task.set_user_pairs(6)
    task.set_up_paths(3)
    task.workload_gen(request_range=(100, 101)),

    # QON.draw(qon.rnet)

    optm = QONOptm(task)
    optm.import_params()
    optm.add_constrs('3456')
    # optm.optimize(ObjType.FEASIBILITY)
    # optm.optimize(ObjType.MAX_THROUGHPUT)
    optm.optimize(ObjType.MIN_LATENCY)




