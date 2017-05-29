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
    codimension = 2;
    tolerance = chi2inv(alpha, codimension);
    
    rand('seed', 0);
    
    % begin iteration
    while (trials < max_trials && consensus_min_cost > threshold)
        % generate three random samples
        sampleIndex = randperm(num_point, 4);
        % compute model H
        H1_inv = fourPoint(point2DHomo1, sampleIndex);
        H2_inv = fourPoint(point2DHomo2, sampleIndex);
        H = H2_inv / H1_inv;
        % compute cost
        cost = 0;
        for i = 1 : num_point
            % compute sampson error
            error_i = sampsonError(H, point2DHomo1(i, :), point2DHomo2(i, :));
            if error_i < tolerance
                cost = cost + error_i;
            else
                cost = cost + tolerance;
            end
        end
        % update model
        if cost < consensus_min_cost
            consensus_min_cost = cost;
            model_H = H;
            % count number of inliers
            dist_error = zeros(1, num_point);
            for i = 1 : num_point
                dist_error(i) = sampsonError(model_H, point2DHomo1(i, :), point2DHomo2(i, :));
            end
            % update max_trials
            num_inliers = sum(dist_error <= tolerance);
            w = num_inliers / num_point;
            max_trials = log(1 - prob) / log(1 - w^4);
        end
            trials = trials + 1;
    end
    % count inliers
    dist_error = zeros(1, num_point);
    for i = 1 : num_point
        dist_error(i) = sampsonError(model_H, point2DHomo1(i, :), point2DHomo2(i, :));
    end
    inlierIndex = (dist_error <= tolerance);
end

% four point method for 2D projective transformation
function H_inv = fourPoint(pointHomo, sampleIndex)
    format longg;
    point1 = pointHomo(sampleIndex(1), :)';
    point2 = pointHomo(sampleIndex(2), :)';
    point3 = pointHomo(sampleIndex(3), :)';
    point4 = pointHomo(sampleIndex(4), :)';
    part1 = [point1, point2, point3];
    lam = part1 \ point4;
    H_inv = [lam(1) * point1, lam(2) * point2, lam(3) * point3];
end

% calculate Sampson error %
function error = sampsonError(H, point1, point2)
    % compute epsilon (Ah) and J
    point1 = point1';
    point2 = point2';
    epsilon = [-point1' * H(2, :)' + point2(2) * point1' * H(3, :)'; ...
               point1' * H(1, :)' - point2(1) * point1' * H(3, :)'];
    J = [-H(2, 1) + point2(2) * H(3, 1), -H(2, 2) + point2(2) * H(3, 2), ...
           0, point1(1) * H(3, 1) + point1(2) * H(3, 2) + H(3, 3); ...
           H(1, 1) - point2(1) * H(3, 1), H(1, 2) - point2(1) * H(3, 2), ...
           -(point1(1) * H(3, 1) + point1(2) * H(3, 2) + H(3, 3)), 0];
    % compute sampson error
    error = epsilon' / (J * J') * epsilon;
end