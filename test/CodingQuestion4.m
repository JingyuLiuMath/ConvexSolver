%% Problem setting.
rng(1);  % For the same result.
p = 300;
n = 500;
c = randn(n, 1);
A = randn(p, n);
x_init = 0.5 * ones(n, 1);
b = A * x_init;

f = @(var_x) c' * var_x;
grad_f = @(var_x) c;
Hessian_f = @(var_x) zeros(n, n);

m = 2 * n;
G = @(var_x) [-var_x; var_x - 1];
diff_G = @(var_x) [-speye(n); speye(n)];
Hessian_G = @(var_x) zeros(n, n, 2 * n);

[x_star, p_star] = linprog(c, [], [], A, b, zeros(n, 1), ones(n, 1));
disp("");
disp("MATLAB linprog");
disp("Check feasibility");
assert(isempty(find(x_star < 0, 1)), "Infeasible solution!")
assert(isempty(find(x_star > 1, 1)), "Infeasible solution!")
disp("norm(A * x_star - b) / norm(b)");
disp(norm(A * x_star - b) / norm(b));
disp("Optimal value");
disp(p_star);
disp("");

%% Barrier method.
x0 = x_init;

t0 = 1;
mu = 10;

tol_feas = 1e-12;
tol = 1e-8;
tol_effective = 1e-6;
max_inner_iter = 200;

alpha = 0.1;
beta = 0.5;

BarrierKKTSolver = @(var_x, var_t) BarrierKKTSolver_Q4(...
    var_x, c, A, b, var_t);
[opt_x, f_opt_x, ...
    outer_iter_number, inner_iter_number, duality_gap_trajectory] ...
    = BarrierFeasibleStart(...
    f, grad_f, Hessian_f, ...
    m, G, diff_G, Hessian_G, ...
    A, b, ...
    x0, t0, mu, ...
    tol_feas, tol, tol_effective, max_inner_iter, ...
    alpha, beta, ...
    BarrierKKTSolver);

disp("");
disp("Barrier method with a feasible starting point");
disp("Check feasibility");
assert(isempty(find(opt_x < 0, 1)), "Infeasible solution!")
assert(isempty(find(opt_x > 1, 1)), "Infeasible solution!")
disp("norm(A * opt_x - b) / norm(b)");
disp(norm(A * opt_x - b) / norm(b));
disp("Optimal value");
disp(f_opt_x);
disp("Outer iteration number");
disp(outer_iter_number);
disp("");

figure;
total_iter_number = sum(inner_iter_number);
duality_gap = [];
for it = 1 : outer_iter_number
    duality_gap = [duality_gap;
        duality_gap_trajectory(it) * ones(inner_iter_number(it), 1)];
end
semilogy(1 : total_iter_number, duality_gap, "LineWidth", 2);
hold on;

title("Duality gap in feasible start barrier method");
xlabel("Iteration number");
ylabel("Duality gap")
saveas(gcf, "duality_gap_barrier.epsc");

%% Primal-dual interior-point method
x0 = 0.8 * rand(n, 1) + 0.1;
lambda0 = -1 ./ G(x0);
nu0 = randn(p, 1);

mu = 10;
tol_feas = 1e-12;
tol = 1e-8;
tol_effective = 1e-6;
max_iter = 200;

alpha = 0.1;
beta = 0.5;

PDKKTSolver = @(var_x, var_lambda, var_nu, var_t) PDKKTSolver_Q4(...
    var_x, var_lambda, var_nu, c, A, b, var_t);
[opt_x, f_opt_x, iter_number, ...
    surrogate_duality_gap_trajectory, ...
    norm_r_pri_trajectory, norm_r_dual_trajectory] ...
    = PrimalDualInteriorPoint(...
    f, grad_f, Hessian_f, ...
    m, G, diff_G, Hessian_G, ...
    A, b, ...
    x0, lambda0, nu0, ...
    mu, tol_feas, tol, tol_effective, max_iter, ...
    alpha, beta, ...
    PDKKTSolver);

disp("");
disp("Primal-dual interior-point method");
disp("Check feasibility");
assert(isempty(find(opt_x < 0, 1)), "Infeasible solution!")
assert(isempty(find(opt_x > 1, 1)), "Infeasible solution!")
disp("norm(A * opt_x - b) / norm(b)");
disp(norm(A * opt_x - b) / norm(b));
disp("Optimal value");
disp(f_opt_x);
disp("Iteration number");
disp(iter_number);
disp("");

figure;
semilogy(0 : iter_number, surrogate_duality_gap_trajectory, "LineWidth", 2);
hold on;

title("Surrogate duality gap in primal-dual interior-point method");
xlabel("Iteration number");
ylabel("Surrogate duality gap");
saveas(gcf, "eta_pdip.epsc");

figure;
semilogy(0 : iter_number, norm_r_pri_trajectory, "LineWidth", 2);
hold on;

title("Norm of primal residual in primal-dual interior-point method");
xlabel("Iteration number");
ylabel("norm(r_pri)")
saveas(gcf, "norm_r_pri_pdip.epsc");

figure;
semilogy(0 : iter_number, norm_r_dual_trajectory, "LineWidth", 2);
hold on;

title("Norm of dual residual in primal-dual interior-point method");
xlabel("Iteration number");
ylabel("norm(r_dual)")
saveas(gcf, "norm_r_dual_pdip.epsc");
