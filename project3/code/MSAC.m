% MSAC method
function [inlierIndex, trials] = MSAC(K, point2DOrig, point3DOrig)
    % transfer to homogeneous
    point2DHomo = [point2DOrig, ones(60, 1)];
    point3DHomo = [point3DOrig, ones(60, 1)];
    
    % calculate the point in normalized coordinates
    point2DNorm = (K \ (point2DHomo'))';

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
        sampleIndex = randperm(60, 3);
        [R, t, cnt] = finsterWalder(point2DNorm, point3DHomo, sampleIndex);
        if cnt == 0
            continue;
        end
        % computation for each possible solution
        for i = 1 : size(R, 3)
            x_cam = K * [R(:, :, i), t(:, :, i)] * point3DHomo';
            x_camHomo = x_cam ./ x_cam(3, :);
            x_cam2D = x_camHomo;
            x_cam2DInhomo = x_cam2D(1 : 2, :);
            error = x_cam2DInhomo - point2DOrig';
            cost = reshape(error, [2*60, 1])' * eye(2*60) * reshape(error, [2*60, 1]);
            if cost < consensus_min_cost
                consensus_min_cost = cost;
                model_R = R(:, :, i);
                model_t = t(:, :, i);
            end
            % count number of inliers
            dist_error = zeros(1, 60);
            for k = 1 : 60
                dist_error(k) = norm(error(:, k));
            end
            num_inliers = sum(dist_error <= tolerance);
            w = num_inliers / 60;
            max_trials = log(1 - prob) / log(1 - w^3);
        end
        trials = trials + 1;
    end

    % count inliers
    x_cam = K * [model_R, model_t] * point3DHomo';
    x_camHomo = x_cam ./ x_cam(3, :);
    x_camInhomo = x_camHomo(1 : 2, :);
    dist_error = zeros(1, 60);
    for k = 1 : 60
        dist_error(k) = norm(error(:, k));
    end
    inlierIndex = (dist_error <= tolerance);
end