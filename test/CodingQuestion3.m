%% Problem setting.
rng(1);  % For the same result.
p = 100;
n = 200;
A = randn(p, n);
x_init = abs(randn(n, 1)) / 2 + 0.01;
b = A * x_init;

f = @(var_x) sum_neg_log(var_x);
grad_f = @(var_x) grad_sum_neg_log(var_x);
Hessian_f = @(var_x) Hessian_sum_neg_log(var_x);

[x_star, p_star] = fmincon(f, x_init, [], [], A, b);
disp("");
disp("MATLAB fmincon");
disp("Check feasibility");
assert(isempty(find(x_star < 0, 1)), "Infeasible solution!");
disp("norm(A * x_star - b) / norm(b)");
disp(norm(A * x_star - b) / norm(b));
disp("Optimal value");
disp(p_star);
disp("");

%% EqConsNewtonfeasibleStart.
Z = null(A);
x0 = x_init + 1e-5 * Z * randn(size(Z, 2), 1);
% x0 = x_init;

tol_feas = 1e-12;
tol = 1e-8;
tol_effective = 1e-5;
max_iter = 200;

alpha = 0.1;
beta = 0.5;

KKTSolver = @(var_x) KKTSolver_Q3(var_x, A, b);

[opt_x, f_opt_x, iter_number, f_x_trajectory] ...
    = EqConsNewtonFeasibleStart(...
    f, grad_f, Hessian_f, ...
    A, b, ...
    x0, ...
    tol_feas, tol, tol_effective, max_iter, ...
    alpha, beta, ...
    KKTSolver);

disp("");
disp("Newton method with a feasible starting point");
disp("Check feasibility");
assert(isempty(find(opt_x < 0, 1)), "Infeasible solution!");
disp("norm(A * opt_x - b) / norm(b)");
disp(norm(A * opt_x - b) / norm(b));
disp("Optimal value");
disp(f_opt_x);
disp("Iteration number");
disp(iter_number);
disp("");

figure(1);
semilogy(0 : iter_number, f_x_trajectory - f_x_trajectory(end), "LineWidth", 2);
title("f(x^{(k)}) - p^{*} in feasible start Newton method");
xlabel("Iteration number");
ylabel("f(x^{(k)}) - p^{*}")
saveas(gcf, "diff_opt_Newton_feasible_start.epsc");

%% EqConsNewtonInfeasibleStart
x0 = abs(randn(n, 1)) + 0.01;
nu0 = randn(p, 1);

tol_feas = 1e-12;
tol = 1e-8;
tol_effective = 1e-5;
max_iter = 200;

alpha = 0.1;
beta = 0.5;

KKTSolver = @(var_x) KKTSolver_Q3(var_x, A, b);

[opt_x, f_opt_x, iter_number, norm_r_x_trajectory] ...
    = EqConsNewtonInfeasibleStart(...
    f, grad_f, Hessian_f, ...
    A, b, ...
    x0, nu0, ...
    tol_feas, tol, tol_effective, max_iter, ...
    alpha, beta, ...
    KKTSolver);

disp("");
disp("Newton method with an infeasible starting point")
disp("Check feasibility")
assert(isempty(find(opt_x < 0, 1)), "Infeasible solution!")
disp("norm(A * opt_x - b) / norm(b)")
disp(norm(A * opt_x - b) / norm(b))
disp("Optimal value");
disp(f_opt_x);
disp("Iteration number");
disp(iter_number);
disp("");

figure(2);
semilogy(0 : iter_number, norm_r_x_trajectory, "LineWidth", 2);
title("Norm of of residual in infeasible start Newton method");
xlabel("Iteration number");
ylabel("norm(r(x^{(k)}, \nu^{(k)}))");
saveas(gcf, "norm_r_Newton_infeasible_start.epsc");

%% Functions used.
function f_x = sum_neg_log(x)

arguments (Input)
    x (:, 1) double;
end

arguments (Output)
    f_x (1, 1) double;
end

if isempty(find(x <= 0, 1))
    f_x = sum(-log(x));
else
    f_x = Inf;
end

end

function grad_f_x = grad_sum_neg_log(x)

arguments (Input)
    x (:, 1) double;
end

arguments (Output)
    grad_f_x (:, 1) double;
end

if isempty(find(x <= 0, 1))
    grad_f_x = -1 ./ x;
else
    grad_f_x = Inf * ones(size(x));
end

end

function Hessian_f_x = Hessian_sum_neg_log(x)

arguments (Input)
    x (:, 1) double;
end

arguments (Output)
    Hessian_f_x (:, :) double;
end

if isempty(find(x <= 0, 1))
    Hessian_f_x = diag(1 ./ (x.^2));
else
    Hessian_f_x = Inf * ones(length(x));
end

end