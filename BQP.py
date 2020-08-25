import cvxopt

from type_checking import *

cvxopt.solvers.options['show_progress'] = False  # silent cvxopt solver


def bqp_cxvopt(H: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
    """
    Solves the following bound-constrained QP:
        min     1/2 x^T H x + b^T x
        s.t.    x >= 0
    :param H: Hessian of the objective function, n by n ndarray
    :param b: vector b, n dimensional ndarray
    extra parameters will be passed to cvxopt.solvers.qp()
    :return: optimal solution x*, an numpy ndarray
    """

    n = H.shape[0]

    sol = cvxopt.solvers.qp(P=cvxopt.matrix(H), q=cvxopt.matrix(b), G=cvxopt.matrix(-np.identity(n)),
                            h=cvxopt.matrix(np.zeros(n)), **kwargs)

    x = np.ravel(sol['x'])

    return x


def bqp_cvxopt_proj(H: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:

    n = H.shape[0]

    sol = cvxopt.solvers.qp(P=cvxopt.matrix(H), q=cvxopt.matrix(b), G=cvxopt.matrix(-np.identity(n)),
                            h=cvxopt.matrix(np.zeros(n)), **kwargs)

    x = np.ravel(sol['x'])

    _, x = cauchy_point(x, H, b)

    return x


def optimality(x: np.ndarray, H: np.ndarray, b: np.ndarray, active: Optional[np.ndarray] = None,
               g: Optional[np.ndarray] = None) -> Tuple[bool, Numeric]:

    if active is None:
        active = x == 0.
    if g is None:
        g = H.dot(x) + b
    active_cond = np.all(g[active] >= 0.)  # gradient components for active constraints must be non-negative
    if np.all(active):
        delta = 0.
    else:
        delta = np.linalg.norm(g[~active], np.inf)

    return active_cond, delta


def cauchy_point(x: np.ndarray, H: np.ndarray, b: np.ndarray,
                 d: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the cauchy point along the negative gradient direction, starting from x,
    assuming the objective function is
        f(x) = .5 * x.dot(H.dot(x)) + b.dot(x)
    with bound constraints
        x >= 0.
    :param x: starting point; n-dimensional array; must be feasible
    :param H: Hessian, symmetric, positive definite matrix; n x n array
    :param b: n-dimensional array
    :param d: direction; if not provided, will compute the negative gradient as the direction, i.e.,
    d = - (H.dot(x) + b)
    :return:
    """

    # assert np.all(x >= 0.), "Infeasible"

    if d is None:
        d = -(H.dot(x) + b)  # negative of the gradient

    T = np.zeros(d.shape[0])

    inf = (d >= 0.)
    normal = ~inf  # normal = (d < 0.)

    T[inf] = +np.inf
    T[normal] = - x[normal] / d[normal]

    TC = set(T[T > 0.])
    TC.add(np.inf)
    TC = sorted(TC)  # breaking points

    t_prev = 0.
    x_prev = x
    for t in TC:
        # search direction
        dt = np.copy(d)
        active = t_prev >= T
        dt[active] = 0.
        x_prev[active] = 0.

        # free = ~active
        # H_reduce = np.copy(H[free, :][:, free])
        # b_reduce = np.copy(b[free])
        # dt_reduce = np.copy(dt[free])
        # x_prev_reduce = np.copy(x_prev[free])

        f1 = b.dot(dt) + x_prev.dot(H.dot(dt))  # directional derivative
        # f1 = b_reduce.dot(dt_reduce) + x_prev_reduce.dot(H_reduce.dot(dt_reduce))
        if f1 >= 0.:  # optimal here
            delta = 0.
        else:
            f2 = dt.dot(H.dot(dt))  # f2 > 0 by assumption
            # f2 = dt_reduce.dot(H_reduce.dot(dt_reduce))
            delta = - f1 / f2  # we must have delta > 0
            assert delta > 0., "Negative curvature"
        t_star = t_prev + delta
        if t_star < t:
            x_star = x_prev + delta * dt
            break
        x_prev = x_prev + (t - t_prev) * dt
        t_prev = t
    else:
        raise RuntimeError

    active = t_star >= T
    x_star[active] = 0.

    active = x_star == 0.

    return active, x_star


def bqp_proj(H: np.ndarray, b: np.ndarray,
             tol: float = 1E-8, max_iter: int = 100) -> np.ndarray:
    n = H.shape[0]
    x_prev = np.zeros(n)
    # active_prev = None
    n_iter = 0

    x = None
    # active = None
    # delta = None

    # f = lambda z: .5 * z.dot(H.dot(z)) + b.dot(z)
    g = H.dot(x_prev) + b

    while True:

        n_iter += 1
        if n_iter > max_iter:
            break

        active, x = cauchy_point(x_prev, H, b, -g)
        # print(x)
        # print(active)

        # trial step
        free = ~active  # free variables
        g = H.dot(x) + b  # gradient

        d_newton = np.zeros(n)
        d_newton[free] = np.linalg.solve(H[free, :][:, free], -g[free])  # subspace minimization

        # f_old = f(x)

        active, x = cauchy_point(x, H, b, d_newton)

        # f_new = f(x)

        # d_newton = np.linalg.solve(H[free, :][:, free], -g[free])  # subspace minimization
        # xt = np.copy(x)
        # xt[free] = np.clip(xt[free] + d_newton, a_min=0., a_max=None)
        # if f(xt) <= f(x):
        #     x = xt
        # else:
        #     idx = d_newton < 0.
        #     if np.any(idx):
        #         alpha_max = min(1., np.min((x[free])[idx] / -d_newton[idx]))
        #     else:
        #         alpha_max = 1.
        #     x[free] = np.clip(x[free] + alpha_max * d_newton, a_min=0., a_max=None)

        # print(alpha_max, f(x) - f(xt))

        # xt = np.clip(xt, a_min=0., a_max=None)  # project back
        # if f(xt) <= f(x):
        #     x = xt  # take trial step
        # xt[free] = cg(xt[free], H[free, :][:, free], g[free])
        # # print(f(x) - f(xt))
        # if f(xt) < f(x):
        #     x = xt

        # delta = np.linalg.norm(x - x_prev, np.inf)
        # print(delta)

        # print("f_old = {:.2E}, f_new = {:.2E}".format(f_old, f_new))

        # active = x == 0.
        g = H.dot(x) + b  # gradient
        if np.all(active):
            delta = 0.
        else:
            delta = np.linalg.norm(g[~active], np.inf)
        # print(np.all(g[active] >= 0.), delta, (np.all(g[active] >= 0.) and delta < tol))
        if np.all(g[active] >= 0.) and delta < tol:
            break

        # active_prev = active
        x_prev = x

    # return {'iter': n_iter, 'x': x, 'active': active, 'delta': delta}
    return x


bqp_solvers = {
        'proj': bqp_proj,
        'cvxopt': bqp_cxvopt,
        'cvxopt_proj': bqp_cvxopt_proj
    }


def decompose(v: np.ndarray, B: np.ndarray, l1: float = 1., l2: float = 1E-5,
              const_term: bool = True, M: Optional[np.ndarray] = None,
              solver: str = 'proj', **kwargs) -> np.ndarray:

    """
    Express v, an n-dimensional ndarray, as non-negative combination of rows of B, an m x n dimensional matrix,
    using regularized least square:
        min_c  .5 * ||B.T c - v||_2^2 + l1 * ||c||_1 + .5 * l2 * ||c||_2^2
        s.t.   c >= 0


    :param v:
    :param B: basis
    :param l1: a non-negative number
    :param l2: a non-negative number
    :param const_term: set to True if the first row of B are all 1s, so the corresponding coefficient
           will NOT be regularized
    :param M: M = B.dot(B.T); if not provided, will be computed
    :param solver: a string which specifies the solver
    :return: c, an m-dimensional ndarray
    """

    if solver not in bqp_solvers:
        raise ValueError('only the following solvers are supported: {}'.format(', '.join(bqp_solvers.keys())))
    else:
        solver_func = bqp_solvers[solver]

    m, n = B.shape

    if isinstance(l1, (int, float)) and l1 >= 0.:
        l1_vec = np.full(shape=m, fill_value=l1)
        if const_term:  # do not regularize const term
            l1_vec[0] = 0.
    else:
        raise ValueError("l1 must be a non-negative number.")

    if isinstance(l2, (int, float)) and l2 >= 0.:
        l2_mat = l2 * np.identity(m)
        if const_term:  # do not regularize const term
            l2_mat[0, 0] = 0.
    else:
        raise ValueError("l2 must be a non-negative number.")

    b = - B.dot(v) + l1_vec

    if M is None:  # compute M
        M = B.dot(B.T)

    H = M + l2_mat

    c = solver_func(H, b, **kwargs)

    return c

