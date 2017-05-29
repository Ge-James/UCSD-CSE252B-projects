% MSAC method
function [inlierIndex, trials] = MSAC(point2DOrig1, point2DOrig2)
    format longg;
    % transfer to homogeneous
    num_point = size(point2DOrig1, 1);
    point2DHomo1 = [point2DOrig1, ones(num_point, 1)];
    point2DHomo2 = [point2DOrig2, ones(num_point, 1)];

    % MSAC algorithm
    consensus_min_cost = Inf;
    trials = 0;
    max_trials = Inf;
    threshold = 0;
    prob = 0.99;
    alpha = 0.95;
    variance = 1;
    codimension = 1;
    tolerance = chi2inv(alpha, codimension);
    
    rand('seed', 2);
    
    % begin iteration
    while (trials < max_trials && consensus_min_cost > threshold)
        % generate seven random samples
        sampleIndex = randperm(num_point, 7);
        % compute model F
        F_sol = sevenPoint(point2DHomo1, point2DHomo2, sampleIndex);
        num_F = size(F_sol, 1) / 3;
        for n = 1 : num_F
            F = F_sol((n - 1) * 3 + 1: n * 3, :);
            % compute cost
            cost = 0;
            for i = 1 : num_point
                % compute sampson error
                error_i = sampsonError(F, point2DHomo1(i, :), point2DHomo2(i, :));
                if error_i < tolerance
                    cost = cost + error_i;
                else
                    cost = cost + tolerance;
                end
            end
            % update model
            if cost < consensus_min_cost
                consensus_min_cost = cost;
                model_F = F;
                % count number of inliers
                dist_error = zeros(1, num_point);
                for i = 1 : num_point
                    dist_error(i) = sampsonError(F, point2DHomo1(i, :), point2DHomo2(i, :));
                end
                % update max_trials
                num_inliers = sum(dist_error <= tolerance);
                w = num_inliers / num_point;
                max_trials = log(1 - prob) / log(1 - w^7);
            end
        end
        trials = trials + 1;
    end
    % count inliers
    dist_error = zeros(1, num_point);
    for i = 1 : num_point
        dist_error(i) = sampsonError(model_F, point2DHomo1(i, :), point2DHomo2(i, :));
    end
    inlierIndex = (dist_error <= tolerance);
end

% seven point method for foundamental matrix
function F_sol = sevenPoint(pointHomo1, pointHomo2, sampleIndex)
    format longg;
    point1 = pointHomo1(sampleIndex, :)';
    point2 = pointHomo2(sampleIndex, :)';
    A = [];
    for i = 1 : 7
        A(i, :) = kron(point2(:, i)', point1(:, i)');
    end
    
    [~,~,V] = svd(A);
    a = V(:,8);
    b = V(:,9);
    F1 = reshape(a,3,3)';
    F2 = reshape(b,3,3)';
    
    syms alph
    F = alph*F1 + F2; 
    equation = solve(det(F));
    F_sol = [];
    alpha_sol = double(vpa(equation));
    for i = 1 : length(alpha_sol)
        alpha = alpha_sol(i);
        if ~isreal(alpha)
            continue;
        end
        F_i = alpha * F1 + F2;
        F_sol = [F_sol; F_i];
    end
end

% calculate Sampson error
function error = sampsonError(F, point1, point2)
    point1 = point1';
    point2 = point2';
    nominator = (point2' * F * point1)^2;
    denominator = (point2' * F(:, 1))^2 + (point2' * F(:, 2))^2 + ...
                  (F(1, :) * point1)^2 + (F(2, :) * point1)^2;
    error = nominator * 1.0 / denominator;
end