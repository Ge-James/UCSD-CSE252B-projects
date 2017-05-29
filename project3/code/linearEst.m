% compute R and t using linear estimation
function [R, t, flag] = linearEst(point2D, point3D)
    format longg
    num = size(point3D, 1);
    
    % calculate mean and covariance of 3D points
    mean3D = mean(point3D);
    covar = cov(point3D);
    [~, D, V] = svd(covar);
    
    % calculate control point in world coordinate
    varWorld = D(1, 1) + D(2, 2) + D(3, 3);
    s = sqrt(varWorld / 3);
    
    % 3D points parameterization
    b = point3D - repmat(mean3D, num, 1);
    alpha2to4 = b * (V / s);
    alpha1 = 1.0 - alpha2to4(:, 1) - alpha2to4(:, 2) - alpha2to4(:, 3);
    alpha = [alpha1, alpha2to4(:, 1), alpha2to4(:, 2), alpha2to4(:, 3)];
    
    % calculage control point in camera coordinate frame
    m = zeros(2 * num, 12);
    for i = 1 : num
        m(2 * i - 1 : 2 * i, :) = ...
            [alpha(i, 1), 0, -alpha(i, 1) * point2D(i, 1), ...
             alpha(i, 2), 0, -alpha(i, 2) * point2D(i, 1), ...
             alpha(i, 3), 0, -alpha(i, 3) * point2D(i, 1), ...
             alpha(i, 4), 0, -alpha(i, 4) * point2D(i, 1); ...
             0, alpha(i, 1), -alpha(i, 1) * point2D(i, 2), ...
             0, alpha(i, 2), -alpha(i, 2) * point2D(i, 2), ...
             0, alpha(i, 3), -alpha(i, 3) * point2D(i, 2), ...
             0, alpha(i, 4), -alpha(i, 4) * point2D(i, 2)];
    end
    [~, ~, V] = svd(m);
    control = V(:, size(V, 2));
    C_cam = [control(1 : 3)'; control(4 : 6)'; control(7 : 9)'; control(10 : 12)'];
    
    % deparameterize 3D point in camera coordinate frame
    X_cam = alpha * C_cam;
    
    % scale 3D points in camera coordinate frame
    varCam = var(X_cam(:, 1)) + var(X_cam(:, 2)) + var(X_cam(:, 3));
    mean_X_cam = mean(X_cam);
    beta = sqrt(varWorld / varCam);
    if mean_X_cam < 0
        beta = -beta;
    end
    X_cam = beta * X_cam;
    
    % transfermation from world coordinate frame to camera coorinate frame
    % calculate R and t based on Umeyama paper
    [R, t, flag] = calCameraPose(point3D, X_cam);
end