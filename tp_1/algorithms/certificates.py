from enum import Enum


class Certificates(Enum):

    FEASIBLE = 'otimo'
    INFEASIBLE = 'inviavel'
    UNBOUNDED = 'ilimitado'
