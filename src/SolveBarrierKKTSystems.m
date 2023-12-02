function [Newton_step_x, w, lambda_square] = SolveBarrierKKTSystems(...
    x, grad_f, Hessian_f, m, G, diff_G, Hessian_G, A, b, t)
% SolveBarrierKKTSystems

% Jingyu Liu, December 1, 2023.

% Solve the following KKT systems
%   [t * Hessian(f)(x) + Hessian(phi)(x), A'; A, 0] * [v; w] = ...
%       -[t * grad(f)(x) + grad(phi)(x); A * x - b]
% at point x where
%   phi(x) = -sum_{i = 1}^m log(-g_i(x)).

% We don't recommend use this default solver.

arguments (Input)
    x (:, 1) double;
    grad_f function_handle;
    Hessian_f function_handle;
    m (1, 1) double
    G function_handle;
    diff_G function_handle;
    Hessian_G function_handle;
    A (:, :) double;
    b (:, 1) double;
    t (:, 1) double;
end

arguments (Output)
    Newton_step_x (:, 1) double;
    w (:, 1) double;
    lambda_square (1, 1) double;
end

G_x = G(x);
diff_G_x = diff_G(x);
Hessian_G_x = Hessian_G(x);
H = t * Hessian_f(x);
gg = t * grad_f(x) - sum(G_x .\ diff_G_x)';
h = A * x - b;

for i = 1 : m
    H = H ...
        + diff_G_x(i, :)' * diff_G_x(i, :) / G_x(i)^2 ...
        - Hessian_G_x(:, :, i) / G_x(i);
end

[p, n] = size(A); 
% KKT_A = [H, A'; A, zeros(p, p)];
% KKT_b = -[gg; h];
% KKT_x = KKT_A \ KKT_b;
% Newton_step_x = KKT_x(1 : n);
% w = KKT_x((n + 1) : end);

HinvAtg = H \ [A', gg];
HinvAt = HinvAtg(:, 1 : p);
Hinvg = HinvAtg(:, p + 1);
S = -A * HinvAt;
w = S \ (A * Hinvg - h);
Newton_step_x = H \ (-A' * w - gg);

lambda_square = Newton_step_x' * H * Newton_step_x;

end