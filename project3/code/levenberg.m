% levenberg iterative method
function [P, w] = levenberg(P_linear, K, inlier2DOrig, inlier3DHomo)
    num_inlier = size(inlier2DOrig, 2);
    % initial estimate
    point2DEstInhomo = projection(K, P_linear, inlier3DHomo);
    error = calMeasureVector((inlier2DOrig - point2DEstInhomo), num_inlier);
    %error = calMeasureVector((inlier2DNorm - point2DEstInhomo), num_inlier);
    covar = eye(2 * num_inlier);
    cost = error' * covar * error;
    disp('cost for each iteration:')
    disp(cost);

    d = diag(inv(K));
    K_diag = diag(repmat(d(1 : 2), [num_inlier, 1]));
    param_num = 6;
    lambda = 0.001;

    paramP = parameterize(P_linear);

    % begin iteration
    for i = 1 : 10
        % compute Jocabian
        J = calJocabian(inlier3DHomo, paramP);
        % compute delta
        delta = (J' * inv(K_diag * covar * K_diag') * J + lambda * eye(param_num)) \...
                (J' * inv(K_diag * covar) * error);
        % update P
        paramPUpdate = paramP + delta;
        deparamPUpdate = deparameterize(paramPUpdate);
        % compute error and cost
        proj2DUpdateInhomo = projection(K, deparamPUpdate, inlier3DHomo);
        errorUpdate = calMeasureVector((inlier2DOrig - proj2DUpdateInhomo), num_inlier);
        % errorUpdate = calMeasureVector((inlier2DNorm - proj2DUpdateInhomo), num_inlier);
        costUpdate = errorUpdate' * covar * errorUpdate;
        disp(costUpdate);
        % make decsion
        if (costUpdate < cost)
            paramP = paramPUpdate;
            deparamP = deparamPUpdate;
            error = errorUpdate;
            cost = costUpdate;
            lambda = 0.1 * lambda;
        else
            lambda = 10.0 * lambda;
        end
    end
    P = deparamP;
    w = paramP(1 : 3);
end

% compute measurement vectort
function measureVector = calMeasureVector(point, num)
    measureVector = reshape(point, [2 * num, 1]);
end

% paramterization
function paramP = parameterize(P)
    R = P(1 : 3, 1 : 3);
    t = P(1 : 3, 4);
    [~, D, V] = svd(R - eye(3));
    v = V(:, 3);
    v_head = [R(3, 2) - R(2, 3);...
              R(1, 3) - R(3, 1);... 
              R(2, 1) - R(1, 2)];

    sin_theta = v' * v_head / 2.0;
    cos_theta = (trace(R) - 1) / 2.0;
    theta = atan2(sin_theta, cos_theta);
    if theta < 0
        theta = theta + 2 * pi;
    end
    w = theta * v / norm(v);
    if theta > pi
        w = w * (1 - 2 * pi / theta * ceil((theta - pi) / (2 * pi)));
    end
    paramP = [w', t']';
end

function deparamP = deparameterize(P)
    w = P(1 : 3);
    t = P(4 : 6);
    theta = norm(w);
    w_x = formMat(w);
    R = cos(theta) * eye(3) + sinc(theta) * w_x + ((1 - cos(theta)) / (theta^2)) * w * w';
    deparamP = [R, t];
end

% sinc(x)
function res = sinc(x)
    if x == 0
        res = 1.0;
    else
        res = (sin(x)) / x;
    end
end

% compute Jocabian matrix
function jocabian = calJocabian(x3D, paramP)
    for i = 1 : size(x3D, 2)
        jocabian(2 * i - 1 : 2 * i, :) = calJocabianAi(paramP, x3D(1 : 3, i));
    end
end

% compute Ai in the Jocabian matrix
function Ai = calJocabianAi(paramP, X)
    w = paramP(1:3);
    theta = norm(w);
    deparamP = deparameterize(paramP);
    xHomo = deparamP * [X; 1];

    part1 = [1/xHomo(3) 0 -xHomo(1)/(xHomo(3)^2); 0 1/xHomo(3) -xHomo(2)/(xHomo(3)^2)];
    part2 = [1, 1, 1, 1, 0, 0;... 
             1, 1, 1, 0, 1, 0;...
             1, 1, 1, 0, 0, 1];
    if theta < eps
        part2(:, 1 : 3) = formMat(-X);
    else
        part2(:, 1 : 3) = sinc(theta) * formMat(-X) + cross(w, X) * (cos(theta) / theta - sin(theta) / (theta^2))...
                          * (w / theta)' + cross(w, cross(w, X)) * (sin(theta) / (theta^2) - 2 * ...
                          (1 - cos(theta)) / (theta^3)) * (w / theta)' + (1 -cos(theta)) / (theta^2)...
                          * eye(3) * (formMat(w) * formMat(-X) + formMat((-cross(w, X))));
    end
    Ai = part1 * part2;
end 

% form the matrix
function mat = formMat(X)
    mat = [0, -X(3), X(2);...
           X(3), 0, -X(1);...
           -X(2), X(1), 0];
end

% projection
function X_est = projection(K, deparamP, point)
    X2DEst = K * deparamP * point;
    X2DEstHomo = X2DEst ./ X2DEst(3, :);
    X2DEstInhomo = X2DEstHomo(1 : 2, :);
    X_est = X2DEstInhomo;
end