function [x, f_x, iter_number, f_x_trajectory] ...
    = EqConsNewtonFeasibleStart(...
    f, grad_f, Hessian_f, ...
    A, b, ...
    x0, ...
    tol_feas, tol, tol_effective, max_iter, ...
    alpha, beta, ...
    KKTSolver)
% EqConsNewtonFeasibleStart

% Jingyu Liu, December 1, 2023.

% Solving the following convex optimization problem with equality
% constraints
%     minimize   f(x)
%    subject to  A * x = b
% by Newton's method with a feasible start point.

% See `SolveKKTSystems` for the interface of the function handle KKTSolver.

arguments (Input)
    f function_handle;
    grad_f function_handle;
    Hessian_f function_handle;
    A (:, :) double;
    b (:, 1) double;
    x0 (:, 1) double;
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
    f_x_trajectory (:, 1) double;
end

% assert(norm(A * x0 - b) <= 1e-12 * norm(b), ...
%     "You must give a feasible start!");

norm_b = norm(b);

x = x0;
f_x = f(x);

f_x_trajectory = f_x;
iter_number = 0;

x_old = Inf * ones(size(x));
f_x_old = Inf;

% Compute Newton step and lambda.
[Newton_step_x, ~, lambda_square_x] = KKTSolver(x);
lambda_square_x0 = lambda_square_x;

cri_lambda_square_x0 = 2 * tol * max(lambda_square_x0, 1);

while iter_number < max_iter ...
        && lambda_square_x >= cri_lambda_square_x0 ...
        && norm(x - x_old) >= tol_effective * max(norm(x_old), 1) ...
        && abs(f_x - f_x_old) >= tol_effective * max(abs(f_x_old), 1)
    x_old = x;
    f_x_old = f_x;

    % Line search and update.
    grad_f_x = grad_f(x);
    [~, x, f_x] = BacktrackingLineSearch(...
        f, x, grad_f_x, Newton_step_x, alpha, beta);

    residual = b - A * x;
    if norm(residual) >= tol_feas * norm_b
        % If not feasible, we must do some refinment.
        x = x + A \ residual;
        f_x = f(x);
    end

    f_x_trajectory = [f_x_trajectory; f_x];
    iter_number = iter_number + 1;

    % Compute Newton step and lambda.
    [Newton_step_x, ~, lambda_square_x] = KKTSolver(x);

end

end