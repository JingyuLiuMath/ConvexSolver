function [x, f_x, iter_number, norm_r_x_trajectory] ...
    = EqConsNewtonInfeasibleStart(...
    f, grad_f, Hessian_f, ...
    A, b, ...
    x0, nu0, ...
    tol_feas, tol, tol_effective, max_iter, ...
    alpha, beta, ...
    KKTSolver)
% EqConsNewtonFeasibleStart

% Jingyu Liu, December 1, 2023.

% Solving the following convex optimization problem with equality
% constraints
%     minimize   f(x)
%    subject to  A * x = b
% by Newton's method with an infeasible start point.

% See `SolveKKTSystems` for the interface of the function handle KKTSolver.

arguments (Input)
    f function_handle;
    grad_f function_handle;
    Hessian_f function_handle;
    A (:, :) double;
    b (:, 1) double;
    x0 (:, 1) double;
    nu0 (:, 1) double = randn(size(b));
    tol_feas (1, 1) double = 1e-12;
    tol (1, 1) double = 1e-8;
    tol_effective (1, 1) double = 1e-6;
    max_iter (1, 1) double = 200;
    alpha (1, 1) double = 0.1;
    beta (1, 1) double = 0.5;
    KKTSolver function_handle ...
        = @(var_x) SolveKKTSystems(var_x, grad_f, Hessian_f, A, b);
end

arguments (Output)
    x (:, 1) double;
    f_x (1, 1) double;
    iter_number (1, 1) double;
    norm_r_x_trajectory (:, 1) double;
end

% r_dual = @(var_x, var_nu) grad_f(var_x) + A' * var_nu;
% r_pri = @(var_x, var_nu) A * var_x - b;
% r = @(var_x, var_nu) [r_dual(var_x, var_nu); r_pri(var_x, var_nu)];

norm_b = norm(b);

x = x0;
nu = nu0;

r_pri = A * x - b;
r_dual = grad_f(x) + A' * nu;
norm_r_pri = norm(r_pri);
% norm_r_pri0 = norm_r_pri;
norm_r_dual = norm(r_dual);
norm_r_dual0 = norm_r_dual;
norm_r = sqrt(norm_r_pri^2 + norm_r_dual^2);
% norm_r0 = norm_r;

norm_r_x_trajectory = norm_r;
iter_number = 0;

cri_norm_r_pri = tol_feas * max(norm_b, 1);
cri_norm_r_dual = tol * max(norm_r_dual0, 1);

x_old = Inf * ones(size(x));

while iter_number < max_iter ...
        && (norm_r_pri >= cri_norm_r_pri ...
        || norm_r_dual >= cri_norm_r_dual) ...
        && norm(x - x_old) >= tol_effective * max(norm(x_old), 1)
    x_old = x;

    % Compute primal and dual Newton step.
    [Newton_step_x, w, ~] = KKTSolver(x);
    Newton_step_nu = w - nu;

    % Backtracking line search on norm(r) and update.
    [~, x, nu, ...
        norm_r_pri, norm_r_dual, norm_r] ...
        = BacktrackingLineSearch_r(x, nu, grad_f, ...
        Newton_step_x, Newton_step_nu, norm_r,...
        A, b, ...
        alpha, beta);

    norm_r_x_trajectory = [norm_r_x_trajectory; norm_r];
    iter_number = iter_number + 1;
end

f_x = f(x);

end

function [step_size, new_x, new_nu, ...
    new_norm_r_pri, new_norm_r_dual, new_norm_r] = BacktrackingLineSearch_r(...
    x, nu, grad_f, dir_x, dir_nu, norm_r,...
    A, b, ...
    alpha, beta)
% BacktrackingLineSearch_r

arguments (Input)
    x (:, 1) double;
    nu (:, 1) double;
    grad_f function_handle;
    dir_x (:, 1) double;
    dir_nu (:, 1) double;
    norm_r (1, 1) double;
    A (:, :) double;
    b (:, 1) double;
    alpha (1, 1) double = 0.1;
    beta (1, 1) double = 0.5;
end

arguments (Output)
    step_size (1, 1) double;
    new_x (:, 1) double;
    new_nu (:, 1) double;
    new_norm_r_pri (1, 1) double;
    new_norm_r_dual (1, 1) double;
    new_norm_r (1, 1) double;
end

step_size = 1;
new_x = x + step_size * dir_x;
new_nu = nu + step_size * dir_nu;
new_r_pri = A * new_x - b;
new_r_dual = grad_f(new_x) + A' * new_nu;
new_norm_r_pri = norm(new_r_pri);
new_norm_r_dual = norm(new_r_dual);
new_norm_r = sqrt(new_norm_r_pri^2 + new_norm_r_dual^2);
while new_norm_r > (1 - alpha * step_size) * norm_r
    step_size = beta * step_size;
    new_x = x + step_size * dir_x;
    new_nu = nu + step_size * dir_nu;
    new_r_pri = A * new_x - b;
    new_r_dual = grad_f(new_x) + A' * new_nu;
    new_norm_r_pri = norm(new_r_pri);
    new_norm_r_dual = norm(new_r_dual);
    new_norm_r = sqrt(new_norm_r_pri^2 + new_norm_r_dual^2);
end

end