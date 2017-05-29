% calculate camera pose R and t based on Umeyama paper
% X is point in world coordinates
% Y is point in camera coodinates
function [R, t, flag] = calCameraPose(X, Y)
    num = size(X, 1);
    mean_X = mean(X, 1);
    mean_Y = mean(Y, 1);
    meanX = repmat(mean_X, [num, 1]);
    meanY = repmat(mean_Y, [num, 1]);
    varX = sum(sum((X - meanX) .* (X - meanX))) / num;
    varY = sum(sum((Y - meanY) .* (Y - meanY))) / num;
    covariance = (Y - meanY)' * (X - meanX) / num;
    
    % if rank is less than, then discard this answer
    if rank(covariance) < size(X, 2) - 1
        flag = false;
    else
        flag = true;
    end
    
    [U, D, V] = svd(covariance);
    S = eye(size(D, 1));
    if abs(det(U) * det(V) - 1) >= 0.01
        S(size(D, 1), size(D, 1)) = -1;
    end
    R = U * S * V';
    c = 1/ varX * trace(D * S);
    t = mean_Y' - c * R * mean_X';
end