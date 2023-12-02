function [x, f_x, ...
    outer_iter_number, inner_iter_number, duality_gap_trajectory] ...
    = BarrierFeasibleStart(...
    f, grad_f, Hessian_f, ...
    m, G, diff_G, Hessian_G, ...
    A, b, ...
    x0, t0, mu, ...
    tol_feas, tol, tol_effective, max_inner_iter, ...
    alpha, beta, ...
    BarrierKKTSolver)
% BarrierFeasibleStart

% Jingyu Liu, December 1, 2023.

% Solving the following convex optimization problem with equality
% constraints
%     minimize   f(x)
%    subject to  g_i(x) <= 0, i = 1, ... , m,
%                A * x = b.
% by barrier method.
% G(x) = [g_1(x); ...; g_m(x)] which is a vector of length m.
% diff_G(x) = [diff(g_1)(x); ...; diff(g_n)(x)] which is an m-by-n matrix.
% Hessian_G(x) = [Hessian(g_1)(x); ...; Hessian(g_1)(x)] which is an
% n-by-n-by-m tensor.

% x0 must satisfies g_x(x0) <= 0, i = 1, ..., m.

% See `SolveBarrierKKTSystems` for the interface of the function handle 
% BarrierKKTSolver.

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
    t0 (1, 1) double = 1;
    mu (1, 1) double = 10;
    tol_feas (1, 1) double = 1e-12;
    tol (1, 1) double = 1e-8;
    tol_effective (1, 1) double = 1e-6;
    max_inner_iter (1, 1) double = 200;
    alpha (1, 1) double = 0.1;
    beta (1, 1) double = 0.5;
    BarrierKKTSolver function_handle ...
        = @(var_x, var_t) SolveBarrierKKTSystems(...
        var_x, grad_f, Hessian_f, m, G, diff_G, Hessian_G, A, b, var_t);
end

arguments (Output)
    x (:, 1) double;
    f_x (1, 1) double;
    outer_iter_number (1, 1) double;
    inner_iter_number (:, 1) double;
    duality_gap_trajectory (:, 1) double;
end

x = x0;
t = t0;

inner_iter_number = [];
duality_gap_trajectory = [];
outer_iter_number = 0;

x_old = x;
f_x_old = f(x);

% Centering step and update.
ft = @(var_x) ft_fun(var_x, f, G, t);
grad_ft = @(var_x) grad_ft_fun(var_x, f, grad_f, G, diff_G, t);
Hessian_ft = @(var_x) Hessian_ft_fun(var_x, f, grad_f, Hessian_f, ...
    G, diff_G, Hessian_G, t);
tmpKKTSolver = @(var_x) BarrierKKTSolver(var_x, t);

[x, ~, inner_iter_number_per_step, ~] = EqConsNewtonFeasibleStart(...
    ft, grad_ft, Hessian_ft, ...
    A, b, ...
    x, ...
    tol_feas, tol, tol_effective, max_inner_iter, ...
    alpha, beta, ...
    tmpKKTSolver);
f_x = f(x);

while m >= tol * t ...
        && norm(x - x_old) >= tol_effective * max(norm(x_old), 1) ...
        && abs(f_x - f_x_old) >= tol_effective * max(abs(f_x_old), 1)
    x_old = x;
    f_x_old = f_x;
    
    % Increase.
    inner_iter_number = [inner_iter_number; inner_iter_number_per_step];
    duality_gap_trajectory = [duality_gap_trajectory; m / t];
    t = mu * t;
    outer_iter_number = outer_iter_number + 1;

    % Centering step and update.
    ft = @(var_x) ft_fun(var_x, f, G, t);
    grad_ft = @(var_x) grad_ft_fun(var_x, f, grad_f, G, diff_G, t);
    Hessian_ft = @(var_x) Hessian_ft_fun(var_x, f, grad_f, Hessian_f, ...
        G, diff_G, Hessian_G, t);
    tmpKKTSolver = @(var_x) BarrierKKTSolver(var_x, t);
    [x, ~, inner_iter_number_per_step, ~] = EqConsNewtonFeasibleStart(...
        ft, grad_ft, Hessian_ft, ...
        A, b, ...
        x, ...
        tol_feas, tol, tol_effective, max_inner_iter, ...
        alpha, beta, ...
        tmpKKTSolver);
    f_x = f(x);
end

end

function ft_x = ft_fun(x, f, G, t)
% ft_fun

arguments (Input)
    x (:, 1) double;
    f function_handle;
    G function_handle;
    t (1, 1) double
end

arguments (Output)
    ft_x (1, 1) double;
end

G_x = G(x);

if isempty(find(G_x >= 0, 1))
    ft_x = t * f(x) - sum(log(-G_x));
else
    ft_x = Inf;
end

end

function grad_ft_x = grad_ft_fun(x, f, grad_f, G, diff_G, t)
% grad_ft_fun

arguments (Input)
    x (:, 1) double;
    f function_handle;
    grad_f function_handle;
    G function_handle;
    diff_G function_handle;
    t (1, 1) double
end

arguments (Output)
    grad_ft_x (:, 1) double;
end

G_x = G(x);

if isempty(find(G_x >= 0, 1))
    diff_G_x = diff_G(x);
    grad_ft_x = t * grad_f(x) - sum(G_x .\ diff_G_x)';
else
    grad_ft_x = Inf * ones(size(x));
end

end

function Hessian_ft_x = Hessian_ft_fun(x, f, grad_f, Hessian_f, ...
    G, diff_G, Hessian_G, t)
% Hessian_ft_fun

arguments (Input)
    x (:, 1) double;
    f function_handle;
    grad_f function_handle;
    Hessian_f function_handle;
    G function_handle;
    diff_G function_handle;
    Hessian_G function_handle;
    t (1, 1) double
end

arguments (Output)
    Hessian_ft_x (:, :) double;
end

G_x = G(x);

if isempty(find(G_x >= 0, 1))
    diff_G_x = diff_G(x);
    Hessian_G_x = Hessian_G(x);
    Hessian_ft_x = t * Hessian_f(x);
    m = length(G_x);
    for i = 1 : m
        Hessian_ft_x = Hessian_ft_x ...
            + diff_G_x(i, :)' * diff_G_x(i, :) / G_x(i)^2 ...
            - Hessian_G_x(:, :, i) / G_x(i);
    end
else
    Hessian_ft_x = Inf * ones(length(x));
end

end