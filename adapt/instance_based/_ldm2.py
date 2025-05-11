import numpy as np
from sklearn.base import check_array
from cvxopt import solvers, matrix
import cvxpy as cp
from numpy.random import seed as set_random_seed

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.metrics import linear_discrepancy
from adapt.utils import set_random_seed

def _frobenius_discrepancy(Xs, Xt):
    m, d = Xt.shape
    M0 = Xt.T @ Xt / m
    diff = M0
    for x in Xs:
        diff -= np.outer(x, x) / len(Xs)
    return np.linalg.norm(diff, "fro")

@make_insert_doc()
class LDM2(BaseAdaptEstimator):
    """
    LDM2 : Linear Discrepancy Minimization
    
    LDM reweights the source instances in order to minimize
    the linear discrepancy between the reweighted source and
    the target data.
    
    The objective function is the following:
    
    .. math::
    
        \min_{||w||_1 = 1, w>0} \max_{||u||=1} |u^T M(w) u|
        
    Where:
    
    - :math:`M(w) = (1/n) X_T^T X_T - X^T_S diag(w) X_S`
    - :math:`X_S, X_T` are respectively the source dataset
      and the target dataset of size :math:`m` and :math:`n`
    
    Parameters
    ----------
    
    Attributes
    ----------
    weights_ : numpy array
        Training instance weights.
    
    estimator_ : object
        Estimator.
    
    See also
    --------
    KMM
    KLIEP
    
    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.instance_based import LDM
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = LDM(RidgeClassifier(), Xt=Xt, random_state=0)
    >>> model.fit(Xs, ys)
    Fit weights...
    Initial Discrepancy : 0.328483
    Final Discrepancy : -0.000000
    Fit Estimator...
    >>> model.score(Xt, yt)
    0.5
    
    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/0902.3430.pdf>`_ \
Y. Mansour, M. Mohri, and A. Rostamizadeh. "Domain \
adaptation: Learning bounds and algorithms". In COLT, 2009.
    """
    
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)
        
        
    def fit_weights(self, Xs, Xt, **kwargs):
        """
        Fit importance weights via Frobenius-norm minimisation (SOCP).

        Parameters
        ----------
        Xs : array-like, shape (n_s, d)
            Source data.

        Xt : array-like, shape (n_t, d)
            Target data.

        kwargs : passed to cvxpy.solve (e.g. max_iters=1e4)

        Returns
        -------
        weights_ : ndarray, shape (n_s,)
            Optimal sample weights.
        """
        # ---- 數據與隨機狀態 ----
        Xs = check_array(Xs)
        Xt = check_array(Xt)
        set_random_seed(self.random_state)

        if self.verbose:
            disc0 = _frobenius_discrepancy(Xs, Xt)
            print(f"Initial Frobenius discrepancy : {disc0:.6f}")

        n_s, d = Xs.shape
        n_t = Xt.shape[0]

        # ---- 構造 M0 及 B 矩陣 ----
        M0 = Xt.T @ Xt / n_t                      # (d × d)
        v0 = M0.ravel()                           # vec(M0) → (d², )

        Mi_vec = [np.outer(x, x).ravel() for x in Xs]   # 每個樣本的 vec(M_i)
        B = np.column_stack(Mi_vec)               # (d² × n_s)

        # ---- 變數 ----
        w = cp.Variable(n_s, nonneg=True)         # 權重 w_i ≥ 0
        t = cp.Variable()                         # 上界 t ≈ ‖resid‖₂

        # ---- 目標與約束 ----
        resid = v0 - B @ w                        # r(w) = vec(M0) - Bw
        constraints = [
            cp.sum(w) == 1,                       # Σ w_i = 1
            cp.norm(resid, 2) <= t                # 二階錐：‖r‖₂ ≤ t
        ]

        solver_kw = {k: v for k, v in kwargs.items()
                    if k in ("max_iters", "eps", "abstol", "reltol", "feastol")}
        problem = cp.Problem(cp.Minimize(t), constraints)
        problem.solve(solver="ECOS", **solver_kw) # 亦可選 OSQP / MOSEK

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"cvxpy failed: {problem.status}")

        # ---- 儲存結果 ----
        w_sdp = np.clip(np.asarray(w.value).ravel(), 0.0, np.inf)
        self.t_ = float(t.value)                  # 最小 Frobenius 範數

        if self.verbose:
            print(f"Final Frobenius discrepancy  : {self.t_:.6f}")
        w_sdp /= w_sdp.sum()
        uniform_alpha = kwargs.pop("uniform_alpha", None)
        if uniform_alpha is not None:
            alpha = float(uniform_alpha)
            # Blend: alpha * uniform + (1-alpha) * w_sdp
            w_mix = alpha * (1.0 / n_s) + (1 - alpha) * w_sdp
            w_mix /= w_mix.sum()
            self.weights_ = w_mix
            if self.verbose:
                print(f"Mixed with uniform (alpha={alpha:.2f})")
        else:
            self.weights_ = w_sdp
            
        return self.weights_
    
    
    def predict_weights(self):
        """
        Return fitted source weights
        
        Returns
        -------
        weights_ : sample weights
        """
        if hasattr(self, "weights_"):
            return self.weights_
        else:
            raise NotFittedError("Weights are not fitted yet, please "
                                 "call 'fit_weights' or 'fit' first.")