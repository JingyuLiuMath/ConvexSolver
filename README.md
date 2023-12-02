# ConvexSolver

ConvexSolver is a MATLAB solver about convex optimization.

## What are included?

For the convex optimization problem with only equality constraints:
$$
\begin{aligned}
\text{minimize} &\quad f(x)\\
\text{subject to}&\quad Ax = b
\end{aligned}
$$
We implement Newton method with either a feasible start or an in feasible start to solve it. See `./src/EqConsNewtonFeasibleStart.m` and `./src/EqConsNewtonInfeasibleStart.m` for more details.

For the general convex optimization problem
$$
\begin{aligned}
\text{minimize} &\quad f(x)\\
\text{subject to} &\quad g_i(x) \leq 0, \ i = 1, \dotsc, m\\
&\quad Ax = b\\
\end{aligned}
$$
We implement barrier method with a feasible start (see `./src/BarrierFeasibleStart.m`) and primal-dual interior-point method (see `./src/PrimalDualInteriorPoint.m`).

We point out in particular that for every method the user could provide a function handle about solving the correspoding KKT systems so that the time can be significantly reduced when the KKT systems can be solved accurate and fast.

## How to use?

Run `ConvexSolver_startup.m`, then change to the `test` directory. The files `CodingQuestion3.m` and `CodingQuestion4.m` are the corresponding code which can be used to show the result in my report.
