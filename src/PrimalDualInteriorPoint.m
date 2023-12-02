function [x, f_x, iter_number, ...
    surrogate_duality_gap_trajectory, norm_r_pri_trajectory, norm_r_dual_trajectory] ...
    = PrimalDualInteriorPoint(...
    f, grad_f, Hessian_f, ...
    m, G, diff_G, Hessian_G, ...
    A, b, ...
    x0, lambda0, nu0, ...
    mu, tol_feas, tol, tol_effective, max_iter, ...
    alpha, beta, ...
    PDKKTSolver)
% PrimalDualInteriorPoint

% Jingyu Liu, December 2, 2023.

% Solving the following convex optimization problem with equality
% constraints
%     minimize   f(x)
%    subject to  g_i(x) <= 0, i = 1, ..., m,
%                A * x = b.
% by primal-dual interior-point method.
% G(x) = [g_1(x); ...; g_m(x)] which is a vector of length m.
% diff_G(x) = [diff(g_1)(x); ...; diff(g_n)(x)] which is an m-by-n matrix.
% Hessian_G(x) = [Hessian(g_1)(x); ...; Hessian(g_1)(x)] which is an
% n-by-n-by-m tensor.

% x0 must satisfies g_x(x0) <= 0, i = 1, ..., m.

% See `SolvePDKKTSystems` for the interface of the function handle
% PDKKTSolver.

arguments (Input)
    f function_handle;
    grad_f function_handle;
    Hessian_f function_handle;
    m (1, 1) double;
    G function_handle;
    diff_G function_handle;
    Hessian_G function_handle;
    A (:, :) double;
    b (:, 1) double;
    x0 (:, 1) double;
    lambda0 (:, 1) double = -1 ./ G(x0);
    nu0 (:, 1) double = randn(size(A, 1), 1);
    mu (1, 1) double = 10;
    tol_feas (1, 1) double = 1e-12;
    tol (1, 1) double = 1e-8;
    tol_effective (1, 1) double = 1e-6;
    max_iter (1, 1) double = 200;
    alpha (1, 1) double = 0.1;
    beta (1, 1) double = 0.5;
    PDKKTSolver function_handle ...
        = @(var_x, var_lambda, var_nu, var_t) SolvePDKKTSystems(...
        var_x, var_lambda, var_nu, ...
        grad_f, Hessian_f, m, G, diff_G, Hessian_G, A, b, var_t);
end

arguments (Output)
    x (:, 1) double;
    f_x (1, 1) double;
    iter_number (1, 1) double;
    surrogate_duality_gap_trajectory (:, 1) double;
    norm_r_pri_trajectory (:, 1) double;
    norm_r_dual_trajectory (:, 1) double;
end

m = length(lambda0);

norm_b = norm(b);
one_vec = ones(m, 1);

x = x0;
lambda = lambda0;
nu = nu0;

G_x = G(x);
eta = -G_x' * lambda;
r_pri = A * x - b;
r_dual = grad_f(x) + diff_G(x)' * lambda + A' * nu;
eta0 = eta;
norm_r_pri = norm(r_pri);
norm_r_dual = norm(r_dual);
norm_r_dual0 = norm_r_dual;

surrogate_duality_gap_trajectory = eta;
norm_r_pri_trajectory = norm_r_pri;
norm_r_dual_trajectory = norm_r_dual;
iter_number = 0;

cri_norm_r_pri = tol_feas * max(norm_b, 1);
cri_norm_r_dual = tol_feas * max(norm_r_dual0, 1);
cri_eta = tol * max(eta0, 1);

x_old = Inf * ones(size(x));
lambda_old = Inf * ones(size(lambda));
nu_old = Inf * ones(size(nu));

while iter_number < max_iter ...
        && (norm_r_pri >= cri_norm_r_pri ...
        || norm_r_dual >= cri_norm_r_dual ...
        || eta >= cri_eta) ...
        && (norm(x - x_old) >= tol_effective * max(norm(x_old), 1) ...
        || norm(lambda - lambda_old) >= tol_effective * max(norm(lambda_old), 1) ...
        || norm(nu - nu_old) >= tol_effective * max(norm(nu_old), 1))
    x_old = x;
    lambda_old = lambda;
    nu_old = nu;

    % Determine t.
    t = mu * m / eta;

    % Compute primal-dual search direction.
    tmpKKTSolver = @(var_x, var_lambda, var_nu) PDKKTSolver(...
        var_x, var_lambda, var_nu, t);
    [dir_x, dir_lambda, dir_nu] = tmpKKTSolver(x, lambda, nu);

    % Backtracking line search on norm(r) and update.
    r_cent = -lambda .* G_x - one_vec / t;
    norm_r_cent = norm(r_cent);
    norm_r = norm([norm_r_pri; norm_r_dual; norm_r_cent]);
    [~, x, lambda, nu, ...
    norm_r_pri, norm_r_dual] = BacktrackingLineSearch_r(...
    x, lambda, nu, ...
    grad_f, ...
    m, G, diff_G, ...
    dir_x, dir_lambda, dir_nu, ...
    norm_r, ...
    A, b, ...
    t, ...
    alpha, beta);

    G_x = G(x);
    eta = - G_x' * lambda;
    surrogate_duality_gap_trajectory ...
        = [surrogate_duality_gap_trajectory; eta];
    norm_r_pri_trajectory = [norm_r_pri_trajectory; norm_r_pri];
    norm_r_dual_trajectory = [norm_r_dual_trajectory; norm_r_dual];
    iter_number = iter_number + 1;
end

f_x = f(x);

end

function [step_size, new_x, new_lambda, new_nu, ...
    new_norm_r_pri, new_norm_r_dual] ...
    = BacktrackingLineSearch_r(...
    x, lambda, nu, ...
    grad_f, ...
    m, G, diff_G, ...
    dir_x, dir_lambda, dir_nu, ...
    norm_r, ...
    A, b, ...
    t, ...
    alpha, beta)
% BacktrackingLineSearch_r

arguments (Input)
    x (:, 1) double;
    lambda (:, 1) double
    nu (:, 1) double;
    grad_f function_handle;
    m (1, 1) double;
    G function_handle;
    diff_G function_handle;
    dir_x (:, 1) double;
    dir_lambda (:, 1) double;
    dir_nu (:, 1) double;
    norm_r (1, 1) double;
    A (:, :) double;
    b (:, 1) double;
    t (1, 1) double;
    alpha (1, 1) double = 0.1;
    beta (1, 1) double = 0.5;
end

arguments (Output)
    step_size (1, 1) double;
    new_x (:, 1) double;
    new_lambda (:, 1) double;
    new_nu (:, 1) double;
    new_norm_r_pri (1, 1) double;
    new_norm_r_dual (1, 1) double;
end

one_vec = ones(m, 1);

id = find(dir_lambda < 0);
if isempty(id)
    s_max = 1;
else
    s_max = min(1, min(-lambda(id) ./ dir_lambda(id)));
end
step_size = 0.99 * s_max;
new_x = x + step_size * dir_x;
new_lambda = lambda + step_size * dir_lambda;
new_nu = nu + step_size * dir_nu;
new_r_pri = A * new_x - b;
new_r_dual = grad_f(new_x) + diff_G(new_x)' * new_lambda + A' * new_nu;
new_r_cent = -lambda .* G(x) - one_vec / t;
new_norm_r_pri = norm(new_r_pri);
new_norm_r_dual = norm(new_r_dual);
new_norm_r_cent = norm(new_r_cent);
new_norm_r = norm([new_norm_r_pri; new_norm_r_dual; new_norm_r_cent]);
while new_norm_r > (1 - alpha * step_size) * norm_r
    step_size = beta * step_size;
    new_x = x + step_size * dir_x;
    new_lambda = lambda + step_size * dir_lambda;
    new_nu = nu + step_size * dir_nu;
    new_r_pri = A * new_x - b;
    new_r_dual = grad_f(new_x) + diff_G(new_x)' * new_lambda + A' * new_nu;
    new_r_cent = - lambda .* G(x) - one_vec / t;
    new_norm_r_pri = norm(new_r_pri);
    new_norm_r_dual = norm(new_r_dual);
    new_norm_r_cent = norm(new_r_cent);
    new_norm_r = norm([new_norm_r_pri; new_norm_r_dual; new_norm_r_cent]);
end

end