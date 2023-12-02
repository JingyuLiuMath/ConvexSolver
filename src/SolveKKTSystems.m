function [Newton_step_x, w, lambda_square] = SolveKKTSystems(...
    x, grad_f, Hessian_f, A, b)
% SolveKKTSystems

% Jingyu Liu, December 1, 2023.

% Solve the following KKT systems
%   [Hessian(f)(x), A'; A, 0] * [v; w] = -[grad(f)(x); A * x - b]
% at point x.

% We don't recommend use this default solver.

arguments (Input)
    x (:, 1) double;
    grad_f function_handle;
    Hessian_f function_handle;
    A (:, :) double;
    b (:, 1) double;
end

arguments (Output)
    Newton_step_x (:, 1) double;
    w (:, 1) double;
    lambda_square (1, 1) double;
end

H = Hessian_f(x);
g = grad_f(x);
h = A * x - b;

[p, n] = size(A); 
% KKT_A = [H, A'; A, zeros(p, p)];
% KKT_b = -[g; h];
% KKT_x = KKT_A \ KKT_b;
% Newton_step_x = KKT_x(1 : n);
% w = KKT_x((n + 1) : end);

HinvAtg = H \ [A', g];
HinvAt = HinvAtg(:, 1 : p);
Hinvg = HinvAtg(:, p + 1);
S = -A * HinvAt;
w = S \ (A * Hinvg - h);
Newton_step_x = H \ (-A' * w - g);

lambda_square = Newton_step_x' * H * Newton_step_x;

end